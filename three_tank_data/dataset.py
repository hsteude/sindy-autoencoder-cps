import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import constants as const
from three_tank_data.data_gen import ThreeTankDataGenerator


class ThreeTankDataSet(Dataset):
    """Write me!"""

    def __init__(self):
        self.df = pd.read_hdf(const.X_SPACE_DATA_PATH)

    def __len__(self):
        """Size of dataset
        """
        return len(self.df)

    def __getitem__(self, index):
        """Get one sample"""
        x = self.df[const.X_COL_NAMES].values[index, :]
        xdot = self.df[const.XDOT_COL_NAMES].values[index, :]
        return x, xdot, index


if __name__ == '__main__':
    # test for lets have a look
    dataset = ThreeTankDataSet()
    idx = 10
    x, xdot, idx = dataset[idx]
    breakpoint()
