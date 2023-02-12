import torch
import gc
from torch import Tensor
from typing import List
from datahandlers.dataset_handler import PreprocessRule
from tqdm import tqdm


# Customized min-max normalization rule
# Accepts a list of 2 2-D Tensors [train, test] (n, f) with the same f size
class NormalizationMinMax(PreprocessRule):

    def __init__(self):
        super(NormalizationMinMax, self).__init__('NormalizationMinMax')

    def preprocess(self, data: List[Tensor]) -> List[Tensor]:
        data_concat = torch.nan_to_num(torch.cat((data[0], data[1]), dim=0))
        processed_data = [torch.nan_to_num(data[0].detach().clone()), torch.nan_to_num(data[1].detach().clone())]
        f = data[0].shape[1]
        print('Normalizing data (min-max)')
        min_list = data_concat.min(0).values
        max_list = data_concat.max(0).values
        del data_concat
        gc.collect()
        for j in tqdm(range(f)):
            min_val = min_list[j]
            max_val = max_list[j]
            diff = max_val - min_val

            processed_data[0][:, j] = (processed_data[0][:, j] - min_val) / diff
            processed_data[1][:, j] = (processed_data[1][:, j] - min_val) / diff

        processed_data[0] = torch.nan_to_num(processed_data[0])
        processed_data[1] = torch.nan_to_num(processed_data[1])

        del min_list, max_list
        gc.collect()

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
        mean_list = data_concat.mean(0)
        std_list = data_concat.std(0)
        del data_concat
        gc.collect()
        for j in tqdm(range(f)):
            mean = mean_list[j]
            std = std_list[j]

            processed_data[0][:, j] = (processed_data[0][:, j] - mean) / std
            processed_data[1][:, j] = (processed_data[1][:, j] - mean) / std

        processed_data[0] = torch.nan_to_num(processed_data[0])
        processed_data[1] = torch.nan_to_num(processed_data[1])

        del mean_list, std_list
        gc.collect()

        return processed_data
