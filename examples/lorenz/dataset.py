import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import examples.lorenz.constants as const



class LorenzBaseDataSet(Dataset):
    """Write me!"""

    def __init__(self):
        path = const.Z_SPACE_DATA_PATH 
        self.df = pd.read_parquet(path)
        self.x = torch.from_numpy(self.df[const.Z_COL_NAMES].values.astype(np.float32))
        self.xdot = torch.from_numpy(self.df[const.Z_DOT_COL_NAMES].values.astype(np.float32))


    def __len__(self):
        """Size of dataset
        """
        return self.x.shape[0] 

    def __getitem__(self, index):
        """Get one sample"""
        return self.x[index, :], self.xdot[index, :], index 

if __name__ == '__main__':
    # test for lets have a look
    dataset = LorenzBaseDataSet()
    idx = 10
    x, xdot, idx = dataset[idx]
    breakpoint() 
