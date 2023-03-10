import pandas as pd
import numpy as np
import os
import random
from typing import List

convert_df = pd.read_csv('../data/DRP2022_preprocessed/ccl_converter_merged.csv')
convert_df = convert_df[['SANGER_ID', 'BROAD_ID']]
convert_df.fillna(-1)
s2b = {}
b2s = {}
count = 0
for _, row in convert_df.iterrows():
    if not pd.isna(row['SANGER_ID']) and not pd.isna(row['BROAD_ID']):
        count += 1
        s = row['SANGER_ID']
        b = row['BROAD_ID']
        if ';' in str(b):
            tmp = str(b).split(';')
            b = tmp[0]
        s2b[s] = b
        b2s[b] = s

gdsc_tuple = pd.read_csv('../data/DRP2022_preprocessed/drug_response/gdsc_tuple_labels_folds.csv')
ctrp_tuple = pd.read_csv('../data/DRP2022_preprocessed/drug_response/ctrp_tuple_labels_folds.csv')
gdsc_ccl_list = list(set(list(gdsc_tuple.loc[:, 'cell_line'])))
ctrp_ccl_list = list(set(list(ctrp_tuple.loc[:, 'cell_line'])))
for i in range(len(gdsc_ccl_list)):
    gdsc_ccl_list[i] = s2b[gdsc_ccl_list[i]]
common_ccl_list2 = list(set(gdsc_ccl_list).intersection(set(ctrp_ccl_list)))
del gdsc_ccl_list, ctrp_ccl_list
common_ccl_list1 = []
for x in common_ccl_list2:
    common_ccl_list1.append(b2s[x])


def remove_ccl_from_tuple(tuple_df: pd.DataFrame, ccl: List) -> pd.DataFrame:
    drop_idxes = []
    for idx, r in tuple_df.iterrows():
        if r['cell_line'] in ccl:
            drop_idxes.append(idx)

    return tuple_df.drop(drop_idxes)


# pure_gdsc_tuple = remove_ccl_from_tuple(gdsc_tuple, common_ccl_list1).drop(columns=['cl_fold', 'pair_fold'])
# pure_ctrp_tuple = remove_ccl_from_tuple(ctrp_tuple, common_ccl_list2).drop(columns=['cl_fold', 'pair_fold'])

# pure_gdsc_tuple.to_csv('../data/DRP2022_preprocessed/drug_response/pure_gdsc_tuple_labels_folds.csv', index=False)
# pure_ctrp_tuple.to_csv('../data/DRP2022_preprocessed/drug_response/pure_ctrp_tuple_labels_folds.csv', index=False)


pure_gdsc_tuple = remove_ccl_from_tuple(gdsc_tuple, common_ccl_list1).drop(columns=['cl_fold'])
packet_lim = int(len(pure_gdsc_tuple) / 5)
cnts = [-1, 0, 0, 0, 0]
for idx, row in pure_gdsc_tuple.iterrows():
    if row['has_expr_from_sanger']:
        fold = random.randint(0, 4)
        while cnts[fold] >= packet_lim:
            fold = random.randint(0, 4)
        pure_gdsc_tuple.at[idx, 'pair_fold'] = fold
        cnts[fold] += 1
    else:
        pure_gdsc_tuple.at[idx, 'pair_fold'] = -1

pure_gdsc_tuple.to_csv('../data/DRP2022_preprocessed/drug_response/resolved_commons/gdsc_tuple_labels_folds.csv',
                       index=False)
