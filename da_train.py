import argparse
import torch
import numpy as np
from models.dann import DANN
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix
from typing import Tuple
from datahandlers.dataset_handler import MyDataset
from os.path import join
from tqdm.auto import tqdm

if torch.cuda.is_available():
    my_device = torch.device('cuda:0')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    my_device = torch.device('mps')
else:
    my_device = torch.device('cpu')

loss_cate = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()
accuracy = Accuracy(task="multiclass", num_classes=2, top_k=1).to(torch.device('cpu'))
conf_mat = ConfusionMatrix(task="binary", num_classes=2).to(torch.device('cpu'))


def load_data_tensor(path: str, batch_size: int, handle_nan: bool = False) -> Tuple[DataLoader, DataLoader, int, int, int]:
    df_tr = torch.load(join(path, 'TRAIN_DF.pt'))
    df_te = torch.load(join(path, 'TEST_DF.pt'))

    # When using drug descriptors, they usually contain nan values after preprocessing
    if handle_nan:
        df_tr = torch.nan_to_num(df_tr)
        df_te = torch.nan_to_num(df_te)
        # In CTRP, [:, 167] or [:, 172] values are extremely small -3.4028e+38, so ignore this drug feature
        df_tr[:, 167] = 0.0
        df_te[:, 167] = 0.0
        df_tr[:, 172] = 0.0
        df_te[:, 172] = 0.0

    mds_tr = MyDataset(torch.load(join(path, 'TRAIN_CCL.pt')),
                       df_tr,
                       torch.load(join(path, 'TRAIN_RESP.pt')),
                       torch.load(join(path, 'TRAIN_DRUGIDX.pt')))

    mds_te = MyDataset(torch.load(join(path, 'TEST_CCL.pt')),
                       df_te,
                       torch.load(join(path, 'TEST_RESP.pt')),
                       torch.load(join(path, 'TEST_DRUGIDX.pt')))

    # Feature sizes, ccl, df, resp
    f1, f2, f3 = mds_tr.get_f_size()

    return DataLoader(mds_tr, batch_size=batch_size, shuffle=False, drop_last=True), \
           DataLoader(mds_te, batch_size=batch_size, shuffle=False, drop_last=True), \
           f1, f2, f3


# y_pred (b, 2) probability, y (b)
def confusion_matrix(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_pred_cate = torch.argmax(y_pred, dim=1)
    return conf_mat(y_pred_cate, y)


def train(source, target, model, optimizer, epoch, epochs):
    s_loader = tqdm(enumerate(source), total=len(source))
    t_loader = tqdm(enumerate(target), total=len(target))
    len_dataloader = min(len(s_loader), len(t_loader))
    loss_total_fin, loss_target_domain_fin, loss_source_y_fin = 0, 0, 0
    y_pred_list, y_list = [], []

    for (i, (xs1, xs2, ys, _)), (_, (xt1, xt2, _, _)) in zip(s_loader, t_loader):
        p = float(i + epoch * len_dataloader) / epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Create domain labels, source (CTRP) [0], target (GDSC) [1]
        s_len = xs1.shape[0]
        ds = torch.zeros(s_len)
        dt = torch.ones(s_len)

        ys, ds, dt = ys.to(torch.int64), ds.to(torch.int64), dt.to(torch.int64)
        xs1, xs2, ys, ds = xs1.to(my_device), xs2.to(my_device), ys.to(my_device), ds.to(my_device)
        xt1, xt2, dt = xt1.to(my_device), xt2.to(my_device), dt.to(my_device)

        model.zero_grad()

        y_pred, domain_pred = model(xs1, xs2, alpha)
        loss_s_y = loss_cate(y_pred, torch.flatten(ys))
        loss_s_domain = loss_domain(domain_pred, ds)

        _, domain_pred = model(xt1, xt2, alpha)
        loss_t_domain = loss_domain(domain_pred, dt)

        loss = loss_s_y + loss_s_domain + loss_t_domain

        loss.backward()
        loss_total_fin += loss.item()
        loss_target_domain_fin += loss_t_domain.item()
        loss_source_y_fin += loss_s_y.item()

        # Record predictions
        y_pred_list.append(y_pred.tolist())
        y_list.append(ys.tolist())

        optimizer.step()

    loss_total_fin = loss_total_fin / len_dataloader
    loss_target_domain_fin = loss_target_domain_fin / len_dataloader
    loss_source_y_fin = loss_source_y_fin / len_dataloader
    y_pred_list = torch.Tensor(y_pred_list).view(-1, 2).to(torch.device('cpu'))
    # Restore to probability
    y_pred_list = torch.exp(y_pred_list)
    y_list = torch.flatten(torch.Tensor(y_list)).to(torch.device('cpu'))
    print(y_pred_list, '\n', y_list)
    print(confusion_matrix(y_pred_list, y_list))
    accuracy_fin = accuracy(y_pred_list, y_list)

    print('EPOCH {} TRAINING SET RESULTS: Average total loss: {:.4f} Average target domain loss: {:.4f}  '
          'Average source response cate loss: {:.4f} Accuracy: {:.4f}'
          .format(epoch, loss_total_fin, loss_target_domain_fin, loss_source_y_fin, accuracy_fin))

    del xs1, xs2, ys, ds, xt1, xt2, dt, y_pred_list, y_list

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def test(target, model, epoch):
    test_loader = tqdm(enumerate(target), total=len(target))
    y_pred_list, y_list = [], []

    for i, (x1, x2, y, _) in test_loader:

        y = y.to(torch.int64)
        x1, x2, y = x1.to(my_device), x2.to(my_device), y.to(my_device)

        y_pred, _ = model(x1, x2, 0)

        # Record predictions
        y_pred_list.append(y_pred.tolist())
        y_list.append(y.tolist())

    y_pred_list = torch.Tensor(y_pred_list).view(-1, 2).to(torch.device('cpu'))
    # Restore to probability
    y_pred_list = torch.exp(y_pred_list)
    y_list = torch.flatten(torch.Tensor(y_list)).to(torch.device('cpu'))
    print(y_pred_list, '\n', y_list)
    print(confusion_matrix(y_pred_list, y_list))
    accuracy_fin = accuracy(y_pred_list, y_list)
    print('EPOCH {} TESTING RESULTS: Accuracy: {:.4f}'
          .format(epoch, accuracy_fin))

    del x1, x2, y, y_pred_list, y_list

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(args):
    source_path = str(args.source_path)
    target_path = str(args.target_path)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    lr = float(args.lr)

    dl_s_tr, dl_s_te, f1, f2, f3 = load_data_tensor(source_path, batch_size, handle_nan=True)
    dl_t_tr, dl_t_te, _, _, _ = load_data_tensor(target_path, batch_size, handle_nan=True)

    model = DANN(f1, f2)
    model = model.to(my_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('Training')
    for epoch in range(1, epochs + 1):
        # source, target
        train(dl_s_tr, dl_t_tr, model, optimizer, epoch, epochs)
        # target
        test(dl_t_te, model, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', default='tensors/DAStandardization/CTRP/pair_fold0', help='Path to the source root')
    parser.add_argument('--target_path', default='tensors/DAStandardization/GDSC/pair_fold0', help='Path to the target root')
    parser.add_argument('--batch_size', default=20, help='Batch size')
    parser.add_argument('--epochs', default=100, help='Total number of epochs')
    parser.add_argument('--lr', default=1e-4, help='Learning rate')
    _args = parser.parse_args()

    main(_args)
