narom examples.lorenz.data_gen import LorenzDataGenerator
import examples.lorenz.constants as const
import pandas as pd


def main():
    lsdg = LorenzDataGenerator(
        number_initial_states=const.NUMBER_INITIAL_STATES,
        number_timesteps=const.NUMBER_TIMESTEPS,
        t_max=const.T_MAX,
        sigma=const.SIGMA,
        beta=const.BETA,
        rho=const.RHO,
        latent_dim=const.LATENT_DIM)
    z, z_dot, time, uid_initial_state = lsdg.generate_z_space_data()
    df_z_space = pd.DataFrame(z, columns=const.Z_COL_NAMES)
    df_z_space[const.Z_DOT_COL_NAMES] = z_dot
    df_z_space[const.TIME_COL_NAME] = time
    df_z_space[const.UID_INITIAL_STATE_COL_NAME] = uid_initial_state

    df_z_space.to_parquet( const.Z_SPACE_DATA_PATH)

if __name__ == '__main__':
    main()
    
