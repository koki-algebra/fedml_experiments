import fedml

from data.UCI import load_data

if __name__ == "__main__":
    args = fedml.init()
    dataset, class_num = load_data(args)

    train_data_num            = dataset[0]
    test_data_num             = dataset[1]
    train_data_global         = dataset[2]
    test_data_global          = dataset[3]
    train_data_local_num_dict = dataset[4]
    train_data_local_dict     = dataset[5]
    test_data_local_dict      = dataset[6]
    class_num                 = dataset[7]

    for batch, (X, y) in enumerate(test_data_local_dict[0]):
        print(f"batch: {batch}, X: {X.size()}")
