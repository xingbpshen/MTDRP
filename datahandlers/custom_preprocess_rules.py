import torch
from torch import Tensor, zeros_like
from typing import List
from datahandlers.dataset_handler import PreprocessRule
from tqdm import tqdm


# Customized min-max normalization rule
# Accepts a list of 2 2-D Tensors [train, test] (n, f) with the same f size
class NormalizationMinMax(PreprocessRule):

    def __init__(self):
        super(NormalizationMinMax, self).__init__('NormalizationMinMax')

    def preprocess(self, data: List[Tensor]) -> List[Tensor]:
        processed_data = [zeros_like(data[0]), zeros_like(data[1])]
        f = data[0].shape[1]
        print('Normalizing data (min-max)')
        for j in tqdm(range(f)):
            min0 = data[0][:, j].min()
            min1 = data[1][:, j].min()
            if min0 < min1:
                min_val = min0
            else:
                min_val = min1

            max0 = data[0][:, j].max()
            max1 = data[1][:, j].max()
            if max0 > max1:
                max_val = max0
            else:
                max_val = max1

            processed_data[0][:, j] = (data[0][:, j] - min_val) / (max_val - min_val)
            processed_data[1][:, j] = (data[1][:, j] - min_val) / (max_val - min_val)

            processed_data[0] = torch.nan_to_num(processed_data[0])
            processed_data[1] = torch.nan_to_num(processed_data[1])

        return processed_data


# Customized standardization rule
# Accepts a list of 2 2-D Tensors [train, test] (n, f) with the same f size
class Standardization(PreprocessRule):

    def __init__(self):
        super(Standardization, self).__init__('Standardization')

    def preprocess(self, data: List[Tensor]) -> List[Tensor]:
        data_concat = torch.nan_to_num(torch.cat((data[0], data[1]), dim=0))
        processed_data = [torch.nan_to_num(data[0].detach().clone()), torch.nan_to_num(data[1].detach().clone())]
        f = data[0].shape[1]
        print('Standardizing data')
        for j in tqdm(range(f)):
            mean = data_concat[:, j].mean()
            std = data_concat[:, j].std()

            processed_data[0][:, j] = (processed_data[0][:, j] - mean) / std
            processed_data[1][:, j] = (processed_data[1][:, j] - mean) / std

        del data_concat

        processed_data[0] = torch.nan_to_num(processed_data[0])
        processed_data[1] = torch.nan_to_num(processed_data[1])

        return processed_data
