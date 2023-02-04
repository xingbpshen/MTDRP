import pandas as pd
import numpy as np
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

    def train_size(self):

        return len(self.ccl_list_tr)

    def test_size(self):

        return len(self.ccl_list_te)

    def train_to_numpy(self):

        return np.array(self.ccl_list_tr), np.array(self.df_list_tr), np.array(self.resp_list_tr)

    def test_to_numpy(self):

        return np.array(self.ccl_list_te), np.array(self.df_list_te), np.array(self.resp_list_te)


class GDSCData:

    def __init__(self, ccl_path, df_path, resp_path):
        self.ccl_df = pd.read_csv(ccl_path)
        self.df_df = pd.read_csv(df_path)
        self.resp_df = pd.read_csv(resp_path)

    def get_fold(self, fold_type, fold_idx):
        fold = OneFold(fold_type, fold_idx)
        print('Parsing GDSC data (fold_type={} fold_idx={})'.format(fold_type, fold_idx))
        for idx, row in tqdm(self.resp_df.iterrows(), total=len(self.resp_df.index)):
            if row['has_expr_from_sanger']:
                ccl = (self.ccl_df[row['cell_line']]).to_numpy()
                df = (self.df_df[self.df_df.iloc[:, 0] == row['drug']]).to_numpy()[0]
                resp = row['ln_ic50']
                row_fold_idx = row[fold_type]
                fold.append_data(ccl, df, resp, row_fold_idx)

        return fold


gdsc = GDSCData('../data/DRP2022_preprocessed/sanger/sanger_broad_ccl_log2tpm.csv',
                '../data/DRP2022_preprocessed/drug_features/gdsc_drug_descriptors.csv',
                '../data/DRP2022_preprocessed/drug_response/gdsc_tuple_labels_folds.csv')
fold_0 = gdsc.get_fold('cl_fold', 0)


exit(0)
