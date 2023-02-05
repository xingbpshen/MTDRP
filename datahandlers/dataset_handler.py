from typing import Tuple
from csv_handler import DRP2022Data, OneFold
from torch.utils.data import Dataset
from torch import Tensor


class PreprocessRule:



class MyDataset(Dataset):

    def __init__(self, ccl: Tensor, df: Tensor, resp: Tensor):
        self.x1 = ccl
        self.x2 = df
        self.y = resp

    def __len__(self) -> int:

        return len(self.y)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:

        return self.x1[idx], self.x2[idx], self.y[idx]


class DRPDataset:


    def __init__(self):
        self.drp2022data = None
        # self.x1_tr = ccl_tr
        # self.x2_tr = df_tr
        # self.y_tr = resp_tr
        # self.x1_te = ccl_te
        # self.x2_te = df_te
        # self.y_te = resp_te

    def load_from_csv(self, source: str, ccl_path: str, df_path: str, resp_path: str):
        self.drp2022data = DRP2022Data(source, ccl_path, df_path, resp_path)

    def get_fold(self, fold_type: str, fold_idx: int, preprocess: PreprocessRule=None) -> Tuple[MyDataset, MyDataset]:
        



# gdsc = DRP2022Data('GDSC',
#                    '../data/DRP2022_preprocessed/sanger/sanger_broad_ccl_log2tpm.csv',
#                    '../data/DRP2022_preprocessed/drug_features/gdsc_drug_descriptors.csv',
#                    '../data/DRP2022_preprocessed/drug_response/gdsc_tuple_labels_folds.csv')
fold_0 = gdsc.get_fold('cl_fold', 0)
tr_ccl, tr_df, tr_resp = fold_0.to_tensor('train')
print(tr_resp)
print(tr_resp.shape)
print(type(tr_resp))
print(tr_resp.min(), tr_resp.max(), tr_resp.mean(), tr_resp.std())

exit(0)
