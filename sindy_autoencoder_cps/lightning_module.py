import torch
from torch import nn
import pytorch_lightning as pl
pl.seed_everything(12354)


class Encoder(nn.Module):
    def __init__(self, input_dim=10000, hidden_size=10000, enc_out_dim=3, *args, **kwargs):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, enc_out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.fc3(out)
        return out

class Decoder(nn.Module):
    def __init__(self, input_dim=3, hidden_size=10000, dec_output_dim=10000, *args, **kwargs):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, dec_output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.fc3(out)
        return out


class SINDyAutoencoder(pl.LightningModule):
    def __init__(self, learning_rate=.001, network_hidden_size=10000, input_dim=10000, 
                 latent_dim=3, number_candidate_functions=10, *args, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.phi_x = Encoder(input_dim=input_dim,
                             hidden_size=network_hidden_size,
                             enc_out_dim=latent_dim)
        self.psi_z = Decoder(input_dim=latent_dim,
                             hidden_size=network_hidden_size,
                             dec_output_dim=input_dim)
        self.XI = torch.zeros(number_candidate_functions, input_dim) 


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def loss_function(self, x, x_hat):
        mse_loss = nn.MSELoss()
        loss = mse_loss(x, x_hat)
        return loss


    def forward(self, x, xdot):
        z = self.phi_x(x)
        return z

    def _shared_eval(self, x):
        z = self.phi_x(x)
        x_hat = self.psi_z(z)
        loss = self.loss_function(x, x_hat) 
        return loss


    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        loss = self._shared_eval(x)
        self.logger.experiment.add_scalars("loss", {'loss_train': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        loss = self._shared_eval(x)
        self.logger.experiment.add_scalars("loss", {'loss_val': loss})
        return loss



