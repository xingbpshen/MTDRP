# DRP 2022
The Drug Response Prediction 2022 project in Computational Biology and Artificial Intelligence (COMBINE) Laboratory, McGill University.

## 1 Setup
### 1.1 Cloning the repository
In the console, type the following command.

    git clone https://github.com/AntonioShen/MTDRP.git

### 1.2 Installing dependencies
It is preferred to use [CONDA](https://conda.io/projects/conda/en/latest/index.html) for dependency packages management.
Type the following command to the console (make sure the current working directory is under the
project root _/MTDRP/_) to create a new environment and to install all required packages.

    conda create --name env --file ./requirements.txt

Activate the newly created CONDA environment.

## 2 Data Preparation
### 2.1 Downloading dataset
Download _DRP2022_preprocesssed.zip_ (not disclosed, will be available in the future), unzip it and merge the folder to 
_./data/DRP2022_preprocessed_.

### 2.2 Parsing .csv files
The dataset contains multiple .csv files, this operation extracts numerical values from them and creates objects (sub-class
of _torch.utils.data.Dataset_) for easy training and testing.

#### 2.2.1 Selecting data folding and the 2nd-stage preprocessing method
A particular set of folds (for cross-validation) with an (optional) addition data preprocessing rule should be determined.
In the example below (see 2.2.3), the first fold (indexed 0) in _cl_fold_ and zero-mean standardization are used to create PyTorch datasets.

It is possible and easy to define a new 2nd-stage preprocessing method in _./datahandlers/custom_preprocess_rules.py_ (see 2.2.2). 
Min-max normalization and zero-mean standardization rules are provided initially.

#### 2.2.2 (Optional) Defining a new 2nd-stage preprocessing method
Every preprocessing method should pack to a class that inherits _datahandlers.dataset_handler.PreprocessRule_, and implements its _preprocess()_
interface to return a list that contains two _torch.Tensor_ for training and testing, respectively.

#### 2.2.3 Example of parsing and saving the first CL fold from the GDSC dataset
In the Python console.

    >>> from datahandlers.dataset_handler import DRPGeneralDataset
    >>> from datahandlers.custom_preprocess_rules import Standardization
    >>> GDSC = DRPGeneralDataset()
    >>> GDSC.load_from_csv('GDSC',
        'data/DRP2022_preprocessed/sanger/sanger_broad_ccl_log2tpm.csv',
        'data/DRP2022_preprocessed/drug_features/gdsc_drug_descriptors.csv',
        'data/DRP2022_preprocessed/drug_response/gdsc_tuple_labels_folds.csv')
    >>> train, test = GDSC.get_fold('cl_fold', 0, preprocess=Standardization(), save=True)
    >>> print(len(train), len(test))
    259386 66319

In the above example, passing _save=True_ saves all tensor files (.pt) under _./tensors/Standardization/GDSC/cl_fold0/_. 
It is recommended to do so.

## 3 Running Experiments
### 3.1 Loading from existing tensor files
In the console, type the following command with arguments _source_path_, _batch_size_, _epochs_ and _lr_ (the learning rate).

    python train.py --source_path ./tensors/Standardization/GDSC/cl_fold0/ --batch_size 20 --epochs 100 --lr 1e-4
