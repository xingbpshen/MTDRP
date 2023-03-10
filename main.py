import argparse
from datahandlers.dataset_handler import DRPGeneralDataset, DRPDADataset
from datahandlers.custom_preprocess_rules import NormalizationMinMax, Standardization

GDSC = DRPGeneralDataset()
GDSC.load_from_csv('GDSC',
                   'data/DRP2022_preprocessed/sanger/sanger_broad_ccl_log2tpm.csv',
                   'data/DRP2022_preprocessed/drug_features/gdsc_drug_descriptors.csv',
                   'data/DRP2022_preprocessed/drug_response/gdsc_tuple_labels_folds.csv')

for i in range(0, 5):
    # Get cl_fold fold 0
    train, test = GDSC.get_fold('pair_fold', i, preprocess=Standardization(), save=True)
    print(len(train), len(test))

CTRP = DRPGeneralDataset()
CTRP.load_from_csv('CTRP',
                   'data/DRP2022_preprocessed/depmap/ccle_log2tpm.csv',
                   'data/DRP2022_preprocessed/drug_features/ctrp_drug_descriptors.csv',
                   'data/DRP2022_preprocessed/drug_response/ctrp_tuple_labels_folds.csv')

for i in range(0, 5):
    # Get cl_fold fold 0
    train, test = CTRP.get_fold('pair_fold', i, preprocess=Standardization(), save=True)
    print(len(train), len(test))

GDSC = DRPDADataset()
GDSC.load_from_csv('GDSC',
                   'data/DRP2022_preprocessed/sanger/sanger_broad_ccl_log2tpm.csv',
                   'data/DRP2022_preprocessed/drug_features/gdsc_drug_descriptors.csv',
                   'data/DRP2022_preprocessed/drug_response/resolved_commons/gdsc_tuple_labels_folds.csv',
                   'data/DRP2022_preprocessed/depmap/ccle_log2tpm.csv',
                   'data/DRP2022_preprocessed/drug_features/ctrp_drug_descriptors.csv')

for i in range(0, 5):
    # Get cl_fold fold 0
    train, test = GDSC.get_fold('pair_fold', i, preprocess=Standardization(), save=True)
    print(len(train), len(test))

CTRP = DRPDADataset()
CTRP.load_from_csv('CTRP',
                   'data/DRP2022_preprocessed/depmap/ccle_log2tpm.csv',
                   'data/DRP2022_preprocessed/drug_features/ctrp_drug_descriptors.csv',
                   'data/DRP2022_preprocessed/drug_response/resolved_commons/ctrp_tuple_labels_folds.csv',
                   'data/DRP2022_preprocessed/sanger/sanger_broad_ccl_log2tpm.csv',
                   'data/DRP2022_preprocessed/drug_features/gdsc_drug_descriptors.csv')

for i in range(0, 5):
    # Get cl_fold fold 0, note that df[:, 147] values -3.4028e+38, need to change to zeros
    train, test = CTRP.get_fold('pair_fold', i, preprocess=Standardization(), save=True)
    print(len(train), len(test))


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--type', help='The type (general, da)')
#     parser.add_argument('--source', help='The source (GDSC, CTRP)')
#     _args = parser.parse_args()
#
#     main(_args)
