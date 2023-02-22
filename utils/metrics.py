import torch
from torch import Tensor
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef
from typing import Tuple

my_device = torch.device('cpu')


class DrugYPair:

    def __init__(self, drug_idx: Tensor):
        self.drug_idx = int(drug_idx)
        self.y_pred_list = []
        self.y_list = []

    def append(self, y_pred: Tensor, y: Tensor):
        self.y_pred_list.append(y_pred.tolist())
        self.y_list.append(y.tolist())

    def get_y_pred(self) -> Tensor:
        return torch.flatten(torch.Tensor(self.y_pred_list))

    def get_y(self) -> Tensor:
        return torch.flatten(torch.Tensor(self.y_list))


class DrugYPairTable:

    def __init__(self, y_preds, ys, drug_idxs):
        self.pairs = []
        for y_pred, y, drug_idx in zip(y_preds, ys, drug_idxs):
            self.append(drug_idx, y_pred, y)

    def append(self, drug_idx: Tensor, y_pred: Tensor, y: Tensor):
        for i in range(len(self.pairs)):
            x = self.pairs[i]
            if x.drug_idx == int(drug_idx):
                x.append(y_pred, y)
                return

        x = DrugYPair(drug_idx)
        x.append(y_pred, y)
        self.pairs.append(x)

    def mean_pcc_per_drug(self) -> Tensor:
        pcc = PearsonCorrCoef().to(my_device)
        len_pairs = len(self.pairs)
        pcc_fin = 0
        for x in self.pairs:
            pcc_fin += pcc(x.get_y_pred(), x.get_y())

        return pcc_fin / len_pairs

    def mean_scc_per_drug(self) -> Tensor:
        scc = SpearmanCorrCoef().to(my_device)
        len_pairs = len(self.pairs)
        scc_fin = 0
        for x in self.pairs:
            scc_fin += scc(x.get_y_pred(), x.get_y())

        return scc_fin / len_pairs


# y_pred (n,), y (n,), drugidx (n,)
def mean_pcc_scc_per_drug(y_pred: Tensor, y: Tensor, drugidx: Tensor) -> Tuple[Tensor, Tensor]:
    y_pred, y, drugidx = y_pred.to(my_device), y.to(my_device), drugidx.to(my_device)
    table = DrugYPairTable(y_pred, y, drugidx)
    return table.mean_pcc_per_drug(), table.mean_scc_per_drug()
