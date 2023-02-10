import argparse
import torch
from models.mlp import ResMLP50 as MLP
from torch.utils.data import DataLoader
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef
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

loss_regression = torch.nn.MSELoss()
pearson = PearsonCorrCoef().to(my_device)
spearman = SpearmanCorrCoef().to(my_device)


def load_data_tensor(path: str, batch_size: int, handle_nan: bool = False) -> Tuple[DataLoader, DataLoader, int, int, int]:
    df_tr = torch.load(join(path, 'TRAIN_DF.pt'))
    df_te = torch.load(join(path, 'TEST_DF.pt'))

    # When using drug descriptors, they usually contain nan values after preprocessing
    if handle_nan:
        df_tr = torch.nan_to_num(df_tr)
        df_te = torch.nan_to_num(df_te)

    mds_tr = MyDataset(torch.load(join(path, 'TRAIN_CCL.pt')),
                       df_tr,
                       torch.load(join(path, 'TRAIN_RESP.pt')))

    mds_te = MyDataset(torch.load(join(path, 'TEST_CCL.pt')),
                       df_te,
                       torch.load(join(path, 'TEST_RESP.pt')))

    # Feature sizes, ccl, df, resp
    f1, f2, f3 = mds_tr.get_f_size()

    return DataLoader(mds_tr, batch_size=batch_size, shuffle=False, drop_last=True), \
           DataLoader(mds_te, batch_size=batch_size, shuffle=False, drop_last=True), \
           f1, f2, f3


def train(dl, model, optimizer, epoch):
    t_loader = tqdm(enumerate(dl), total=len(dl))
    len_dataloader = len(t_loader)
    mse_fin = 0
    y_pred_list, y_list = [], []

    for i, (x1, x2, y) in t_loader:

        if epoch % 20 == 0:
            for s in optimizer.param_groups:
                s['lr'] = s['lr'] / 10

        x1, x2, y = x1.to(my_device), x2.to(my_device), y.to(my_device)

        model.zero_grad()

        y_pred = model(x1, x2)

        loss = loss_regression(y_pred, y)
        y_pred_list.append(torch.flatten(y_pred).tolist())
        y_list.append(torch.flatten(y).tolist())

        loss.backward()

        mse_fin += loss.item()

        optimizer.step()

    mse_fin = mse_fin / len_dataloader
    y_pred_list = torch.flatten(torch.Tensor(y_pred_list)).to(my_device)
    y_list = torch.flatten(torch.Tensor(y_list)).to(my_device)
    print(y_pred_list, '\n', y_list)
    pearson_fin = pearson(y_pred_list, y_list)
    spearman_fin = spearman(y_pred_list, y_list)
    print('EPOCH {} TRAINING SET RESULTS: Average regression loss: {:.4f} pearson: {:.4f} spearman: {:.4f}'
          .format(epoch, mse_fin, pearson_fin, spearman_fin))

    del x1, x2, y, y_pred_list, y_list

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def test(dl, model, epoch):
    test_loader = tqdm(enumerate(dl), total=len(dl))
    len_dataloader = len(test_loader)
    mse_fin = 0
    y_pred_list, y_list = [], []

    for i, (x1, x2, y) in test_loader:

        x1, x2, y = x1.to(my_device), x2.to(my_device), y.to(my_device)

        y_pred = model(x1, x2)

        loss = loss_regression(y_pred, y)
        y_pred_list.append(torch.flatten(y_pred).tolist())
        y_list.append(torch.flatten(y).tolist())

        mse_fin += loss.item()

    mse_fin = mse_fin / len_dataloader
    y_pred_list = torch.flatten(torch.Tensor(y_pred_list)).to(my_device)
    y_list = torch.flatten(torch.Tensor(y_list)).to(my_device)
    print(y_pred_list, '\n', y_list)
    pearson_fin = pearson(y_pred_list, y_list)
    spearman_fin = spearman(y_pred_list, y_list)
    print('EPOCH {} TESTING RESULTS: Average mse: {:.4f} pearson: {:.4f} spearman: {:.4f}'
          .format(epoch, mse_fin, pearson_fin, spearman_fin))

    del x1, x2, y, y_pred_list, y_list

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(args):
    source_path = str(args.source_path)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    lr = float(args.lr)

    dl_tr, dl_te, f1, f2, f3 = load_data_tensor(source_path, batch_size, handle_nan=True)

    model = MLP(f1, f2, f3)
    model = model.to(my_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('Training')
    for epoch in range(1, epochs + 1):
        train(dl_tr, model, optimizer, epoch)
        test(dl_te, model, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', help='Path to the data root')
    parser.add_argument('--batch_size', default=20, help='Batch size')
    parser.add_argument('--epochs', default=100, help='Total number of epochs')
    parser.add_argument('--lr', default=1e-4, help='Learning rate')
    _args = parser.parse_args()

    main(_args)
