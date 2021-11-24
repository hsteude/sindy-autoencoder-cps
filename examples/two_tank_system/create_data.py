from examples.two_tank_system.data_gen import TwoTankDataGenerator
import examples.two_tank_system.constants as const
import pandas as pd


def main():
    ttdg = TwoTankDataGenerator(
        number_initial_states=const.NUMBER_INITIAL_STATES,
        number_timesteps=const.NUMBER_TIMESTEPS,
        t_max=const.T_MAX,
        q1=const.Q1, q3=const.Q3, A=const.A, g=const.G,
        latent_dim=const.LATENT_DIM)

    x, x_dot, z, z_dot, time, uid_initial_state = ttdg.generate_x_space_data()

    df_x = pd.DataFrame(x, columns=const.X_COL_NAMES)
    df_xdot = pd.DataFrame(x_dot, columns=const.XDOT_COL_NAMES)
    df = pd.concat((df_x, df_xdot), axis=1)
    # df[const.Z_COL_NAMES] = z
    df[const.TIME_COL_NAME] = time
    df[const.UID_INITIAL_STATE_COL_NAME] = uid_initial_state
    df[const.Z_DOT_COL_NAMES] = z_dot
    df.to_parquet(const.X_SPACE_DATA_PATH)

    #save smaller version for DEBUGGING
    df_small =  df.iloc[0:1000, :]
    df_small.to_parquet(const.X_SPACE_DATA_PATH_DEBUGGING)

if __name__ == '__main__':
    main()
    
