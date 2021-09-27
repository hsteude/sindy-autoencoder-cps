# data generation related
NUMBER_TIMESTEPS = 101
NUMBER_INITIAL_STATES = 1000
T_MAX = 10
Q1 = 0
Q3 = 0
A = 10
G = 9.81
LATENT_DIM = 3

PICTURE_SIZE = 100
H1_X_RANGE = (0, 32)
H2_X_RANGE = (34, 66)
H3_X_RANGE = (68, 100)
BLUR_FULTER_SIGMA = 0

INITIAL_LEVEL_MIN = 10
INITIAL_LEVEL_MAX = 90

Z_COL_NAMES = ['h1', 'h2', 'h3']
Z_DOT_COL_NAMES = ['h1_dot', 'h2_dot', 'h3_dot']
TIME_COL_NAME = 'time'
UID_INITIAL_STATE_COL_NAME = 'uid_initial_state'

#path
X_SPACE_DATA_PATH = 'data/processed/x_space_data.parquet'
X_SPACE_DATA_PATH_DEBUGGING = 'data/processed/x_space_data_debugging.parquet'
Z_SPACE_DATA_PATH = 'data/processed/z_space_data.parquet'
X_COL_NAMES = [f'x_{i}' for i in range(PICTURE_SIZE**2)]
XDOT_COL_NAMES = [f'xdot_{i}' for i in range(PICTURE_SIZE**2)]

SEED = 12354

