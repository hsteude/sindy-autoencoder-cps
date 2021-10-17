"""
...
"""
import numpy as np
from scipy.integrate import odeint
import examples.lorenz.constants as const
# from scipy.ndimage import gaussian_filter

np.random.seed(62654)


class LorenzDataGenerator():

    def __init__(self, number_initial_states=2,
                 number_timesteps=const.NUMBER_TIMESTEPS, t_max=const.T_MAX,
                 sigma=const.SIGMA, rho=const.RHO, beta=const.BETA,
                 latent_dim=3,
                 derivatives=False):
        self.t_max = t_max
        self.number_timesteps = number_timesteps
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.t = np.linspace(0, self.t_max, self.number_timesteps)
        self.latent_dim = latent_dim
        self.number_initial_states = number_initial_states
        self.initial_states = self.get_random_inition_states(N=number_initial_states)
        self.derivatives = derivatives

    def system_dynamics_function(self, X, t):
        x = X[0]
        y = X[1]
        z = X[2]
        dx_dt = self.sigma * (y-x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x*y - self.beta*z

        return dx_dt, dy_dt, dz_dt


    def get_random_inition_states(self, N):
        initial_states = np.array(np.random.uniform(low=const.INITIAL_STATE_MIN,
                                                    high=const.INITIAL_STATE_MAX,
                                                    size=N*self.latent_dim))
        return initial_states.reshape(N, self.latent_dim)

    def solve_ode(self, initial_state):
        return odeint(self.system_dynamics_function, initial_state, self.t)

    def compute_derivatives(self, x, dt):
        """
        First order forward difference (forward difference)
        TODO: Find out how the pysindy authors, came up with the formula for the start and end points
        """
        # Uniform timestep (assume t contains dt)
        x_dot = np.full_like(x, fill_value=np.nan)
        if np.isscalar(dt):
            x_dot[:-1, :] = (x[1:, :] - x[:-1, :]) / dt
            x_dot[-1, :] = (3 * x[-1, :] / 2 - 2 * x[-2, :] + x[-3, :] / 2) / dt 
        return x_dot

    def generate_z_space_data(self):
        z = np.zeros((self.number_timesteps * self.number_initial_states, self.latent_dim))
        z_dot = np.zeros((self.number_timesteps * self.number_initial_states, self.latent_dim))
        time = np.array(list(self.t)*self.number_initial_states)
        uid_initial_state = np.array([[i]*self.number_timesteps
                                      for i in range(self.number_initial_states)]).ravel()
        for i in range(self.number_initial_states):
            z_i = self.solve_ode(self.initial_states[i,:])
            z_dot_i =  self.compute_derivatives(z_i, dt=self.t_max / (self.number_timesteps - 1))
            start_idx = i*self.number_timesteps
            end_idx = i*self.number_timesteps+self.number_timesteps
            z[start_idx:end_idx, :] = z_i
            z_dot[start_idx:end_idx, :] = z_dot_i
        return z, z_dot, time, uid_initial_state


if __name__ == '__main__':
    lsdg = LorenzDataGenerator()
    z, z_dot, time, uid_initial_state = lsdg.generate_z_space_data()


