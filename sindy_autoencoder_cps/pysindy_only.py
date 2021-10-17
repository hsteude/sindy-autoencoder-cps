import torch
from torch import nn
import pytorch_lightning as pl
from sindy_autoencoder_cps.sindy_library import SINDyLibrary
pl.seed_everything(12354)



class SINDy(pl.LightningModule):
    def __init__(self,
                 learning_rate=.001,
                 latent_dim=3,
                 sindy_biases=True,
                 sindy_states=False,
                 sindy_sin=False,
                 sindy_cos=False,
                 sindy_multiply_pairs=True,
                 sindy_poly_order=3,
                 sindy_sqrt=False,
                 sindy_inverse=False,
                 sindy_sign_sqrt_of_diff=True,
                 sequential_thresholding=True,
                 sequential_thresholding_freq = 10,
                 sequential_thresholding_thres = 1e-4,
                 loss_weight_sindy_x = 1,
                 loss_weight_sindy_z = 1,
                 loss_weight_sindy_regularization = 1,
                 *args, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.SINDyLibrary = SINDyLibrary(
                 device='cuda:0',
                 latent_dim=latent_dim,
                 include_biases=sindy_biases,
                 include_states=sindy_states,
                 include_sin=sindy_sin,
                 include_cos=sindy_cos,
                 include_multiply_pairs=sindy_multiply_pairs,
                 poly_order=sindy_poly_order,
                 include_sqrt=sindy_sqrt,
                 include_inverse=sindy_inverse,
                 include_sign_sqrt_of_diff=sindy_sign_sqrt_of_diff)


        self.XI = nn.Parameter(torch.full((self.SINDyLibrary.number_candidate_functions, latent_dim), .1))

        self.sequential_thresholding = sequential_thresholding
        self.sequential_thresholding_freq = sequential_thresholding_freq
        self.sequential_thresholding_thres = sequential_thresholding_thres
        # TODO: here we got a device mess!
        self.XI_coefficient_mask = torch.ones((self.SINDyLibrary.number_candidate_functions, latent_dim),
                                              device='cuda:0')

        self.loss_weight_sindy_z = loss_weight_sindy_z
        self.loss_weight_sindy_regularization = loss_weight_sindy_regularization
        self.loss_names = ['total_loss', 'sindy_loss_z', 'sindy_regular_loss']

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)

    def loss_function(self, zdot, zdot_hat, XI):
        mse = nn.MSELoss()
        sindy_loss_z = mse(zdot, zdot_hat)
        sindy_regular_loss = torch.sum(torch.abs(XI))
        loss = self.loss_weight_sindy_z * sindy_loss_z \
            + self.loss_weight_sindy_regularization * sindy_regular_loss 
        return loss, sindy_loss_z, sindy_regular_loss


    def forward(self, x, xdot):
        return self._shared_eval(x, xdot)

    def _shared_eval(self, z, zdot):
        theta = self.SINDyLibrary.transform(z)
        if self.sequential_thresholding:
            zdot_hat = torch.matmul(theta, self.XI_coefficient_mask * self.XI)
        else:
            zdot_hat = torch.matmul(theta, self.XI)
        return z, zdot, zdot_hat


    def training_step(self, batch, batch_idx):
        z, zdot, _ = batch
        z, zdot, zdot_hat = self._shared_eval(z, zdot)
        losses = self.loss_function(zdot=zdot, zdot_hat=zdot_hat, XI=self.XI) 
        loss_dict = dict(zip([f'train_{n}' for n in self.loss_names], losses))
        self.logger.experiment.add_scalars("loss", loss_dict)
        return loss_dict['train_total_loss']

    def validation_step(self, batch, batch_idx):
        z, zdot, _ = batch
        z, zdot, zdot_hat = self._shared_eval(z, zdot)
        losses = self.loss_function(zdot=zdot, zdot_hat=zdot_hat, XI=self.XI) 
        loss_dict = dict(zip([f'val_{n}' for n in self.loss_names], losses))
        self.logger.experiment.add_scalars("loss", loss_dict)
        return loss_dict['val_total_loss']



