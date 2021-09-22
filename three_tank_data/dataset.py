import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import constants as const
from three_tank_data.data_gen import ThreeTankDataGenerator


class ThreeTankDataSet(Dataset):
    """Write me!"""

    def __init__(self, number_initial_states=100,
                 number_timesteps=101, t_max=10,
                 q1=0, q3=0, A=10, g=9.81, latent_dim=3):
        ttdg = ThreeTankDataGenerator(
            number_initial_states=number_initial_states,
            number_timesteps=number_timesteps,
            t_max=t_max,
            q1=q1, q3=q3, A=A, g=g,
            latent_dim=latent_dim)

        self.x, self.x_dot, self.time, self.uid_initial_state \
            = ttdg.generate_x_space_data()

    def __len__(self):
        """Size of dataset
        """
        return self.x.shape[0]

    def __getitem__(self, index):
        """Get one sample"""
        TOOOOOODOOOOOOOOOOOO, hier war ich gestern
        out = self.df_data[self.df_data.sample_idx == index][['signal_1', 'signal_2']]\
            .values.astype(np.float32)
        return out, index


if __name__ == '__main__':
    # test for lets have a look
    dataset = SimpleRandomCurvesDataset(data_path=const.DATA_PATH,
                                        hidden_states_path=const.HIDDEN_STATE_PATH)
    idx = 10
    ts = dataset[idx]
    breakpoint()
    print(type(ts))
