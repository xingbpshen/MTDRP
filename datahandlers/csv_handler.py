import pandas as pd
import numpy as np
import torch
import gc
from tqdm import tqdm
from typing import Tuple
from torch import Tensor


class OneFold:

    def __init__(self, fold_type, fold_idx):
        self.fold_type = fold_type
        self.fold_idx = fold_idx
        # Splitting to training and testing sets
        self.ccl_list_tr, self.ccl_list_te = [], []
        self.df_list_tr, self.df_list_te = [], []
        self.resp_list_tr, self.resp_list_te = [], []

    # Appending one row of data
    def append_data(self, ccl, df, resp, fold_idx):
        if fold_idx == self.fold_idx and int(fold_idx) >= 0:
            self.ccl_list_te.append(ccl)
            self.df_list_te.append(df)
            self.resp_list_te.append(resp)
        elif fold_idx != self.fold_idx and int(fold_idx) >= 0:
            self.ccl_list_tr.append(ccl)
            self.df_list_tr.append(df)
            self.resp_list_tr.append(resp)
        else:
            print('Error in append_data, invalid fold_idx.')
            exit(1)

    def __len__(self):

        return len(self.ccl_list_tr) + len(self.ccl_list_te)

    def get_samples_quantity(self, usage: str) -> int:
        if str(usage).lower() == 'train':
            return len(self.ccl_list_tr)
        elif str(usage).lower() == 'test':
            return len(self.ccl_list_te)
        else:
            print('Invalid usage in get_samples_quantity()')
            exit(1)

    def to_numpy(self, usage: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if str(usage).lower() == 'train':
            return np.array(self.ccl_list_tr, dtype=float), np.array(self.df_list_tr, dtype=float), \
                   np.array(self.resp_list_tr, dtype=float)
        elif str(usage).lower() == 'test':
            return np.array(self.ccl_list_te, dtype=float), np.array(self.df_list_te, dtype=float), \
                   np.array(self.resp_list_te, dtype=float)
        else:
            print('Invalid usage in to_numpy()')
            exit(1)

    def to_tensor(self, usage: str) -> Tuple[Tensor, Tensor, Tensor]:
        ccl, df, resp = self.to_numpy(usage)

        return torch.from_numpy(ccl).type(torch.float32), torch.from_numpy(df).type(torch.float32), torch.from_numpy(
            resp).type(torch.float32).view(-1, 1)


class DRP2022Data:

    def __init__(self, source: str, ccl_path: str, df_path: str, resp_path: str):
        self.source = str(source).upper()
        if self.source != 'GDSC' and self.source != 'CTRP':
            print('Invalid data source')
            exit(1)
        self.ccl_df = pd.read_csv(ccl_path)
        self.df_df = pd.read_csv(df_path)
        self.resp_df = pd.read_csv(resp_path)

    def get_fold(self, fold_type: str, fold_idx: int) -> OneFold:
        fold = OneFold(fold_type, fold_idx)
        print('Parsing {} data (fold_type={}, fold_idx={})'.format(self.source, fold_type, fold_idx))

        if self.source == 'GDSC':
            for idx, row in tqdm(self.resp_df.iterrows(), total=len(self.resp_df.index)):
                if row['has_expr_from_sanger']:
                    ccl = (self.ccl_df[row['cell_line']]).to_numpy()
                    df = (self.df_df[self.df_df.iloc[:, 0] == row['drug']]).iloc[:, 1:].to_numpy()[0]
                    resp = row['ln_ic50']
                    row_fold_idx = row[fold_type]
                    fold.append_data(ccl, df, resp, row_fold_idx)

        elif self.source == 'CTRP':
            for idx, row in tqdm(self.resp_df.iterrows(), total=len(self.resp_df.index)):
                if row['has_expr_from_depmap']:
                    ccl = (self.ccl_df[row['cell_line']]).to_numpy()
                    df = (self.df_df[self.df_df.iloc[:, 0] == row['drug']]).iloc[:, 1:].to_numpy()[0]
                    resp = row['auc']
                    row_fold_idx = row[fold_type]
                    fold.append_data(ccl, df, resp, row_fold_idx)

        return fold


class DRP2022DAData(DRP2022Data):

    def __init__(self, source: str, ccl_path: str, df_path: str, resp_path: str, ccl2_path: str, df2_path: str):
        super(DRP2022DAData, self).__init__(source, ccl_path, df_path, resp_path)

        self.ccl_df = pd.read_csv(ccl_path, index_col=0).T
        ccl2_df = pd.read_csv(ccl2_path, index_col=0).T
        common_genes = list(set(self.ccl_df.columns).intersection(set(ccl2_df.columns)))
        self.ccl_df = self.ccl_df[common_genes]
        del ccl2_df
        # Get intersection genes
        self.ccl_df = self.ccl_df.T.reset_index()

        self.df_df = pd.read_csv(df_path, index_col=0)
        df2_df = pd.read_csv(df2_path, index_col=0)
        common_descriptors = list(set(self.df_df.columns).intersection(set(df2_df.columns)))
        self.df_df = self.df_df[common_descriptors]
        del df2_df
        self.df_df = self.df_df.reset_index()

        gc.collect()

    def get_fold(self, fold_type: str, fold_idx: int) -> OneFold:
        fold = OneFold(fold_type, fold_idx)
        print('Parsing {} data (fold_type={}, fold_idx={}) for domain adaptation'.format(self.source, fold_type, fold_idx))

        def pre_run() -> DrugRespTable:
            print('Pre-running')
            _table = DrugRespTable()
            if self.source == 'GDSC':
                for _idx, _row in tqdm(self.resp_df.iterrows(), total=len(self.resp_df.index)):
                    if _row['has_expr_from_sanger']:
                        _resp = _row['ln_ic50']
                        _table.append(_row['drug'], float(_resp))
                return _table

            elif self.source == 'CTRP':
                for _idx, _row in tqdm(self.resp_df.iterrows(), total=len(self.resp_df.index)):
                    if _row['has_expr_from_depmap']:
                        _resp = _row['auc']
                        _table.append(_row['drug'], float(_resp))
                return _table

        table = pre_run()

        if self.source == 'GDSC':
            for idx, row in tqdm(self.resp_df.iterrows(), total=len(self.resp_df.index)):
                if row['has_expr_from_sanger']:
                    ccl = (self.ccl_df[row['cell_line']]).to_numpy()
                    df = (self.df_df[self.df_df.iloc[:, 0] == row['drug']]).iloc[:, 1:].to_numpy()[0]
                    resp = row['ln_ic50']
                    resp_mean = table.get_resp_mean(row['drug'])
                    if float(resp) >= resp_mean:
                        cate = 1
                    else:
                        cate = 0
                    row_fold_idx = row[fold_type]
                    fold.append_data(ccl, df, cate, row_fold_idx)

        elif self.source == 'CTRP':
            for idx, row in tqdm(self.resp_df.iterrows(), total=len(self.resp_df.index)):
                if row['has_expr_from_depmap']:
                    ccl = (self.ccl_df[row['cell_line']]).to_numpy()
                    df = (self.df_df[self.df_df.iloc[:, 0] == row['drug']]).iloc[:, 1:].to_numpy()[0]
                    resp = row['auc']
                    resp_mean = table.get_resp_mean(row['drug'])
                    if float(resp) >= resp_mean:
                        cate = 1
                    else:
                        cate = 0
                    row_fold_idx = row[fold_type]
                    fold.append_data(ccl, df, cate, row_fold_idx)

        del table
        gc.collect()

        return fold


class DrugRespPair:

    def __init__(self, drug: str):
        self.drug = drug
        self.resp = []

    def append(self, resp: float):
        self.resp.append(float(resp))


class DrugRespTable:

    def __init__(self):
        self.list = []

    def append(self, drug: str, resp: float):
        for i in range(len(self.list)):
            if self.list[i].drug == drug:
                self.list[i].append(resp)
                return

        new_pair = DrugRespPair(drug)
        new_pair.append(resp)
        self.list.append(new_pair)

    def get_resp_mean(self, drug: str) -> float:
        for x in self.list:
            if x.drug == drug:
                resp = np.array(x.resp)
                return resp.mean()

        print('Error in get_resp_mean()')
        exit(1)
