import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import constants as const
from three_tank_data.data_gen import ThreeTankDataGenerator


class ThreeTankDataSet(Dataset):
    """Write me!"""

    def __init__(self, debug=False):
        path = const.X_SPACE_DATA_PATH_DEBUGGING if debug else const.X_SPACE_DATA_PATH
        self.df = pd.read_parquet(path)
        self.x = torch.from_numpy(self.df[const.X_COL_NAMES].values.astype(np.float32))
        self.xdot = torch.from_numpy(self.df[const.XDOT_COL_NAMES].values.astype(np.float32))


    def __len__(self):
        """Size of dataset
        """
        return self.x.shape[0] 

    def __getitem__(self, index):
        """Get one sample"""
        return self.x[index, :], self.xdot[index, :], index 


if __name__ == '__main__':
    # test for lets have a look
    dataset = ThreeTankDataSet()
    idx = 10
    x, xdot, idx = dataset[idx]
    breakpoint()
