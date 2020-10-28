import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import datatable as dt
from utils.print_decorator import print_if_complete
from config import cfg


@print_if_complete
def save_as_jay(file_name):
    """ Creates a copy of the raw training dataset in CSV format to one in Jay format
    """
    cfg.TRAIN.FILE_NAME = file_name
    dt.fread(cfg.DATA_DIR + cfg.TRAIN.FILE_NAME +
             ".csv").to_jay(cfg.DATA_DIR + cfg.TRAIN.FILE_NAME + ".jay")


@print_if_complete
def readData():
    """ Returns Panda DataFrames of selected files.
    """
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    sample_df = pd.read_csv('data/sample_submission.csv')

    return train_df, test_df, sample_df


@print_if_complete
def split_train_val(train_df, val_ratio=0.005):
    val_df = train_df.iloc[int((1-val_ratio)*len(train_df)):]
    train_df = train_df.iloc[: int((1-val_ratio)*len(train_df))]
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df