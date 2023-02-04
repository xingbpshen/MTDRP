import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


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

    def get_samples_quantity(self, usage):
        if str(usage).lower() == 'train':
            return len(self.ccl_list_tr)
        elif str(usage).lower() == 'test':
            return len(self.ccl_list_te)
        else:
            print('Invalid usage in get_samples_quantity()')
            exit(1)

    def to_numpy(self, usage):
        if str(usage).lower() == 'train':
            return np.array(self.ccl_list_tr, dtype=float), np.array(self.df_list_tr, dtype=float), \
                   np.array(self.resp_list_tr, dtype=float)
        elif str(usage).lower() == 'test':
            return np.array(self.ccl_list_te, dtype=float), np.array(self.df_list_te, dtype=float), \
                   np.array(self.resp_list_te, dtype=float)
        else:
            print('Invalid usage in to_numpy()')
            exit(1)

    def to_tensor(self, usage):
        ccl, df, resp = self.to_numpy(usage)

        return torch.from_numpy(ccl).type(torch.float32), torch.from_numpy(df).type(torch.float32), torch.from_numpy(
            resp).type(torch.float32)


class DRP2022Data:

    def __init__(self, source, ccl_path, df_path, resp_path):
        self.source = str(source).upper()
        if self.source != 'GDSC' and self.source != 'CTRP':
            print('Invalid data source')
            exit(1)
        self.ccl_df = pd.read_csv(ccl_path)
        self.df_df = pd.read_csv(df_path)
        self.resp_df = pd.read_csv(resp_path)

    def get_fold(self, fold_type, fold_idx):
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
                break

        elif self.source == 'CTRP':
            for idx, row in tqdm(self.resp_df.iterrows(), total=len(self.resp_df.index)):
                if row['has_expr_from_depmap']:
                    ccl = (self.ccl_df[row['cell_line']]).to_numpy()
                    df = (self.df_df[self.df_df.iloc[:, 0] == row['drug']]).iloc[:, 1:].to_numpy()[0]
                    resp = row['auc']
                    row_fold_idx = row[fold_type]
                    fold.append_data(ccl, df, resp, row_fold_idx)

        return fold


gdsc = DRP2022Data('GDSC',
                   '../data/DRP2022_preprocessed/sanger/sanger_broad_ccl_log2tpm.csv',
                   '../data/DRP2022_preprocessed/drug_features/gdsc_drug_descriptors.csv',
                   '../data/DRP2022_preprocessed/drug_response/gdsc_tuple_labels_folds.csv')
fold_0 = gdsc.get_fold('cl_fold', 0)
tr_ccl, _, _ = fold_0.to_tensor('train')
print(tr_ccl)
print(tr_ccl.shape)
print(type(tr_ccl))
print(tr_ccl.min(), tr_ccl.max(), tr_ccl.mean(), tr_ccl.std())

exit(0)
