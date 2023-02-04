import pandas as pd
import numpy as np


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
        if fold_idx == self.fold_idx and fold_idx >= 0:
            self.ccl_list_te.append(ccl)
            self.df_list_te.append(df)
            self.resp_list_te.append(resp)
        elif fold_idx != self.fold_idx and fold_idx >= 0:
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


class DRPData:

    def __init__(self, ccl_path, df_path, resp_path):
        self.ccl_df = pd.read_csv(ccl_path)
        self.df_df = pd.read_csv(df_path)
        self.resp_df = pd.read_csv(resp_path)

    def get_fold(self, fold_type, fold_idx):


def load_data_for_fold(ccl_path, df_path, resp_path, fold_type, fold_idx):
    df_ccl = pd.read_csv(ccl_path)
    df_df = pd.read_csv(df_path)
    df_resp = pd.read_csv(resp_path)



load_ccl_features_responses('../data/DRP2022_preprocessed/sanger/sanger_broad_ccl_log2tpm.csv',
                            '../data/DRP2022_preprocessed/drug_response/gdsc_tuple_labels_folds.csv',
                            '../data/DRP2022_preprocessed/drug_features/gdsc_drug_descriptors.csv')

exit(0)
