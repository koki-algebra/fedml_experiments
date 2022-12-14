import math
from typing import Tuple, Dict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset, DataLoader
import fedml
from fedml.arguments import Arguments

from utils import one_hot_encoding, normalize


class UCIIncome(Dataset):
    def __init__(self, X: FloatTensor, y: LongTensor = None) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index) -> Tuple[FloatTensor, LongTensor] | FloatTensor:
        x = self.X[index]
        if self.y is None:
            return x
        else:
            y = self.y[index]
            return x, y

    @staticmethod
    def get_dataset(is_norm = True, train_size = 0.8, labeled_size = 0.1) -> Tuple[FloatTensor, LongTensor, FloatTensor, FloatTensor, LongTensor]:
        target_column_name = "salary"

        # read csv
        parent = Path(__file__).resolve().parent
        df: pd.DataFrame = pd.read_csv(parent.joinpath("data/adult.csv"), sep=",")

        # one-hot encoding
        df = one_hot_encoding(df, target_column_name)

        # normalization
        if is_norm:
            df = normalize(df)

        # data split
        train_data: np.ndarray
        test_data:  np.ndarray
        train_l:    np.ndarray
        train_u:    np.ndarray
        train_data, test_data = train_test_split(df.values, train_size=train_size)
        train_l, train_u = train_test_split(train_data, train_size=labeled_size)

        # transform ndarray to tensor
        X_l_train = torch.from_numpy(train_l[:,:-1].astype(np.float32)).clone()
        y_train = torch.from_numpy(train_l[:,-1].astype(np.int64)).clone()
        X_u_train = torch.from_numpy(train_u[:,:-1].astype(np.float32)).clone()
        X_test    = torch.from_numpy(test_data[:,:-1].astype(np.float32)).clone()
        y_test    = torch.from_numpy(test_data[:,-1].astype(np.int64)).clone()

        return X_l_train, y_train, X_u_train, X_test, y_test


def load_data(args: Arguments) -> Tuple[list, int]:
    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)

    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data(args)

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]

    return dataset, class_num


def load_partition_data(args: Arguments) -> Tuple[int, int, int, list, list, dict, dict, dict, int]:
    labeled_batch_size = args.labeled_batch_size
    
    # get dataset
    X_l_train, y_train, X_u_train, X_test, y_test = UCIIncome.get_dataset()
    train_labeled_dataset   = UCIIncome(X=X_l_train, y=y_train)
    train_unlabeled_dataset = UCIIncome(X=X_u_train)
    test_dataset            = UCIIncome(X=X_test, y=y_test)

    client_number  = args.client_num_in_total
    train_data_num = len(X_l_train) + len(X_u_train)
    test_data_num  = len(X_test)
    train_data_global = DataLoader(dataset=train_labeled_dataset, batch_size=labeled_batch_size)
    test_data_global  = DataLoader(dataset=test_dataset, batch_size=labeled_batch_size)
    train_data_local_num_dict: Dict[int, int] = {}
    train_data_local_dict: Dict[int, DataLoader] = {}
    test_data_local_dict : Dict[int, DataLoader] = {}
    class_num = 2

    # create client local data
    client_train_data_num = math.floor(train_data_num / client_number)
    client_test_data_num  = math.floor(test_data_num  / client_number)
    for i in range(client_number):
        train_local_dataset = UCIIncome(
            X=X_l_train[i*client_train_data_num:(i+1)*client_train_data_num],
            y=y_train[i*client_train_data_num:(i+1)*client_train_data_num]
        )
        test_local_dataset = UCIIncome(
            X=X_test[i*client_test_data_num:(i+1)*client_test_data_num],
            y=y_test[i*client_test_data_num:(i+1)*client_test_data_num]
        )
        train_data_local_dict[i] = DataLoader(
            dataset=train_local_dataset,
            batch_size=labeled_batch_size
        )
        test_data_local_dict[i]  = DataLoader(
            dataset=test_local_dataset,
            batch_size=labeled_batch_size
        )

        train_data_local_num_dict[i] = client_train_data_num

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num
    )
