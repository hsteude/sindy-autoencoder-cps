# LORENZ CONSTANTS
NUMBER_TIMESTEPS = 10001
NUMBER_INITIAL_STATES = 100
T_MAX = 100
SIGMA = 10
RHO = 28
BETA = 8 / 3
INITIAL_STATE_MIN = .01
INITIAL_STATE_MAX = .1
LATENT_DIM = 3

Z_COL_NAMES = ['x', 'y', 'z']
Z_DOT_COL_NAMES = ['x_dot', 'y_dot', 'z_dot']
TIME_COL_NAME = 'time'
UID_INITIAL_STATE_COL_NAME = 'uid_initial_state'

#path
X_SPACE_DATA_PATH = 'data/processed/lorenz_x_space_data.parquet'
X_SPACE_DATA_PATH_DEBUGGING = 'data/processed/lorenz_x_space_data_debugging.parquet'
Z_SPACE_DATA_PATH = 'data/processed/lorenz_z_space_data.parquet'

SEED = 12354
