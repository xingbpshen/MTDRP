import os
from abc import ABC, abstractmethod
from typing import Tuple, List
from datahandlers.csv_handler import DRP2022Data
from torch.utils.data import Dataset
from torch import Tensor
from torch import save as torch_save


# Implement it in custom_preprocess_rule.py
class PreprocessRule(ABC):

    def __init__(self, tag: str):
        self.tag = tag

    # Input as a list of [train, test], output as [train, test]
    @abstractmethod
    def preprocess(self, data: List[Tensor]) -> List[Tensor]:
        pass


class MyDataset(Dataset):

    def __init__(self, ccl: Tensor, df: Tensor, resp: Tensor):
        self.x1 = ccl
        self.x2 = df
        self.y = resp

    def __len__(self) -> int:

        return self.x1.shape[0]

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:

        return self.x1[idx], self.x2[idx], self.y[idx]

    def get_f_size(self) -> Tuple[int, int, int]:

        return self.x1.shape[1], self.x2.shape[1], self.y.shape[1]


class DRPGeneralDataset:

    def __init__(self):
        self.source = None
        self.__drp2022data: DRP2022Data = None

    def load_from_csv(self, source: str, ccl_path: str, df_path: str, resp_path: str):
        self.__drp2022data = DRP2022Data(source, ccl_path, df_path, resp_path)
        self.source = source.upper()

    # Returns train and test datasets
    def get_fold(self, fold_type: str, fold_idx: int,
                 preprocess: PreprocessRule = None, save=True) -> Tuple[MyDataset, MyDataset]:
        if self.__drp2022data is None:
            print('Please load the data first in DRPDataset')
            exit(1)
        else:
            one_fold = self.__drp2022data.get_fold(fold_type, fold_idx)
            print('Transferring to torch.Tensor')
            ccl_tr, df_tr, resp_tr = one_fold.to_tensor('train')
            ccl_te, df_te, resp_te = one_fold.to_tensor('test')
            if preprocess is not None:
                print('Preprocessing')
                ccl_tmp = preprocess.preprocess([ccl_tr, ccl_te])
                df_tmp = preprocess.preprocess([df_tr, df_te])
                resp_tmp = preprocess.preprocess([resp_tr, resp_te])

                if save:
                    save_path = os.path.join(os.getcwd(), 'tensors', preprocess.tag, self.source,
                                             fold_type.lower() + str(fold_idx))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    print('Saving processed torch.Tensor to {}'.format(save_path))
                    torch_save(ccl_tmp[0], os.path.join(save_path, 'TRAIN_CCL.pt'))
                    torch_save(ccl_tmp[1], os.path.join(save_path, 'TEST_CCL.pt'))
                    torch_save(df_tmp[0], os.path.join(save_path, 'TRAIN_DF.pt'))
                    torch_save(df_tmp[1], os.path.join(save_path, 'TEST_DF.pt'))
                    torch_save(resp_tmp[0], os.path.join(save_path, 'TRAIN_RESP.pt'))
                    torch_save(resp_tmp[1], os.path.join(save_path, 'TEST_RESP.pt'))

                return MyDataset(ccl_tmp[0], df_tmp[0], resp_tmp[0]), MyDataset(ccl_tmp[1], df_tmp[1], resp_tmp[1])

            else:

                if save:
                    save_path = os.path.join(os.getcwd(), 'tensors', 'raw', self.source,
                                             fold_type.lower() + str(fold_idx))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    print('Saving processed torch.Tensor to {}'.format(save_path))
                    torch_save(ccl_tr, os.path.join(save_path, 'TRAIN_CCL.pt'))
                    torch_save(ccl_te, os.path.join(save_path, 'TEST_CCL.pt'))
                    torch_save(df_tr, os.path.join(save_path, 'TRAIN_DF.pt'))
                    torch_save(df_te, os.path.join(save_path, 'TEST_DF.pt'))
                    torch_save(resp_tr, os.path.join(save_path, 'TRAIN_RESP.pt'))
                    torch_save(resp_te, os.path.join(save_path, 'TEST_RESP.pt'))

                return MyDataset(ccl_tr, df_tr, resp_tr), MyDataset(ccl_te, df_te, resp_te)
