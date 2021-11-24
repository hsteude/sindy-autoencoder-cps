from examples.two_tank_system.data_module import TwoTankDataModule
from sindy_autoencoder_cps.lightning_module import SINDyAutoencoder
import torch
import pytorch_lightning as pl
from sindy_autoencoder_cps.callbacks import SequentialThresholdingCallback 


HPARAMS = dict(
    learning_rate=1e-4,
    input_dim=6, 
    latent_dim=2,
    enc_hidden_sizes=[512, 128, 64, 32],
    dec_hidden_sizes=[32, 64, 128, 512],
    activation='tanh',
    debug=False,
    validdation_split=.1,
    batch_size=32,
    dl_num_workers=24,
    max_epochs=10_000,
    sindy_biases=True,
    sindy_states=True,
    sindy_sin=True,
    sindy_cos=True,
    sindy_multiply_pairs=True,
    sindy_poly_order=2,
    sindy_sqrt=False,
    sindy_inverse=False,
    sindy_sign_sqrt_of_diff=True,
    sequential_thresholding=True,
    sequential_thresholding_freq = 100,
    sequential_thresholding_thres = 0.005,
    loss_weight_sindy_x=1e2,
    loss_weight_sindy_z=1e-0,
    loss_weight_sindy_regularization=1e-3,
)

def train():
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        gradient_clip_val=0.1,
        gpus=gpus,
        max_epochs=10_000,
        precision=64,
        # stochastic_weight_avg=True,
        callbacks=[SequentialThresholdingCallback()])
    model = SINDyAutoencoder(**HPARAMS)
    dm = TwoTankDataModule(validdation_split=HPARAMS['validdation_split'],
                             batch_size=HPARAMS['batch_size'],
                             dl_num_workers=HPARAMS['dl_num_workers'],
                             debug=HPARAMS['debug'])
    trainer.fit(model, dm)

if __name__ == '__main__':
    # continue_training()
    train()
