from typing import List
from datahandlers.dataset_handler import PreprocessRule
from torch import Tensor, zeros_like
from tqdm import tqdm


# Customized min-max normalization rule
# Accepts a list of 2 2-D Tensors (n, f) with the same f size
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

        return processed_data


# def test(preprocess: PreprocessRule = None):
#     t1 = Tensor([[1, 2], [2, 1], [4, 4]])
#     t2 = Tensor([[2, 1], [1, 2]])
#     preprocess.preprocess([t1, t2])
#
#
# test(NormalizationMinMax())
