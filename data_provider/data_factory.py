
# mahdis's notes:
# This file acts as a "factory" that creates the right dataset
# and dataloader based on the command-line arguments.

# Usage: data_provider(args, flag) returns (dataset, dataloader)
#   - args: command-line arguments (dataset name, batch size, etc.)
#   - flag: 'train', 'val', or 'test'


from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader

# Dictionary mapping dataset names to their corresponding classes.
# When the user specifies --data custom, it maps to Dataset_Custom.
data_dict = {
    'ETTh1': Dataset_ETT_hour,      # Hourly ETT data (2-hour interval)
    'ETTh2': Dataset_ETT_hour,      # Hourly ETT data (2-hour interval)
    'ETTm1': Dataset_ETT_minute,    # 15-minute ETT data
    'ETTm2': Dataset_ETT_minute,    # 15-minute ETT data
    'custom': Dataset_Custom,       # Custom dataset (Electricity, Traffic, Weather, etc.)
}


def data_provider(args, flag):
    """
    Creates and returns a dataset and dataloader for training/validation/testing.

    Args:
        args: Command-line arguments containing dataset settings.
        flag: 'train', 'val', or 'test' to load the appropriate split.

    Returns:
        data_set: The dataset object (contains the actual data).
        data_loader: PyTorch DataLoader for batching during training.
    """

    # Look up the dataset class based on args.data
    Data = data_dict[args.data]

    # Determine time feature encoding:
    #   0 = simple features (month, day, weekday, hour)
    #   1 = advanced features from time_features()
    timeenc = 0 if args.embed != 'timeF' else 1

    # Shuffle training data for better generalization, but NOT test data
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True

    # Keep all samples in the last batch
    drop_last = False

    # Extract key parameters from args
    batch_size = args.batch_size
    freq = args.freq

    # Create the dataset instance
    # This loads the data, scales it (if enabled), and creates sliding windows
    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )

    # Print the number of samples in this split
    print(flag, len(data_set))

    # Create a PyTorch DataLoader for efficient batching
    # num_workers=0 means data loading happens in the main process (faster on CPU)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader