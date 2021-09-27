from three_tank_data.data_module import ThreeTankDataModule
from sindy_autoencoder_cps.lightning_module import SINDyAutoencoder
import torch
import pytorch_lightning as pl


HPARAMS = dict(
    learning_rate=1e-4,
    network_hidden_size=1000,
    input_dim=10000, 
    latent_dim=3,
    activation='tanh',
    debug=False,
    number_candidate_functions=10,
    validdation_split=.1,
    batch_size=256,
    dl_num_workers=24,
    max_epochs=10_000,
    sindy_biases=True,
    sindy_states=True,
    sindy_sin=False,
    sindy_cos=False,
    sindy_multiply_pairs=True,
    sindy_poly_order=2,
    sindy_sqrt=False,
    sindy_inverse=False,
    sindy_sign_sqrt_of_diff=True,
    sequential_thresholding=True,
    loss_weight_sindy_x=1e-2,
    loss_weight_sindy_z=1e-8,
    loss_weight_sindy_regularization=0,
)

def train():
        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=10_000)
        model = SINDyAutoencoder(**HPARAMS)
        dm = ThreeTankDataModule(validdation_split=HPARAMS['validdation_split'],
                                 batch_size=HPARAMS['batch_size'],
                                 dl_num_workers=HPARAMS['dl_num_workers'],
                                 debug=HPARAMS['debug'])
        trainer.fit(model, dm)



if __name__ == '__main__':
    # continue_training()
    train()
