from three_tank_data.data_module import ThreeTankDataModule
from sindy_autoencoder_cps.lightning_module import SINDyAutoencoder
import torch
import pytorch_lightning as pl
from sindy_autoencoder_cps.callbacks import SequentialThresholdingCallback 


HPARAMS = dict(
    learning_rate=1e-4,
    network_hidden_size=1000,
    input_dim=10000, 
    latent_dim=3,
    enc_hidden_sizes=[1028, 512, 128, 64],
    dec_hidden_sizes=[64, 128, 512, 1028],
    activation='tanh',
    debug=False,
    number_candidate_functions=10,
    validdation_split=.1,
    batch_size=32,
    dl_num_workers=24,
    max_epochs=10_000,
    sindy_biases=False,
    sindy_states=False,
    sindy_sin=False,
    sindy_cos=False,
    sindy_multiply_pairs=False,
    sindy_poly_order=3,
    sindy_sqrt=False,
    sindy_inverse=False,
    sindy_sign_sqrt_of_diff=True,
    sequential_thresholding=True,
    sequential_thresholding_freq = 10,
    sequential_thresholding_thres = 0.01,
    loss_weight_sindy_x=1e-3,
    loss_weight_sindy_z=1e-5,
    loss_weight_sindy_regularization=1e-5,
)

def train():
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        gradient_clip_val=0.1,
        gpus=gpus,
        max_epochs=10_000,
        stochastic_weight_avg=True,
        callbacks=[SequentialThresholdingCallback()])
    model = SINDyAutoencoder(**HPARAMS)
    dm = ThreeTankDataModule(validdation_split=HPARAMS['validdation_split'],
                             batch_size=HPARAMS['batch_size'],
                             dl_num_workers=HPARAMS['dl_num_workers'],
                             debug=HPARAMS['debug'])
    trainer.fit(model, dm)

def continue_training():
    LAST_CKP = 'lightning_logs/version_11/checkpoints/epoch=240-step=85795.ckpt'
    model = SINDyAutoencoder.load_from_checkpoint(LAST_CKP, **HPARAMS)
    trainer = pl.Trainer(resume_from_checkpoint=LAST_CKP, max_epochs=100000, gpus=1)
    dm = ThreeTankDataModule(validdation_split=HPARAMS['validdation_split'],
                             batch_size=HPARAMS['batch_size'],
                             dl_num_workers=HPARAMS['dl_num_workers'],
                             debug=HPARAMS['debug'])
    trainer.fit(model, dm)




if __name__ == '__main__':
    # continue_training()
    train()
