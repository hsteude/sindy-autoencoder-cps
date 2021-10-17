from examples.three_tank_system.data_module import ThreeTankDataModule
from sindy_autoencoder_cps.lightning_module import SINDyAutoencoder
import torch
import pytorch_lightning as pl
from sindy_autoencoder_cps.callbacks import SequentialThresholdingCallback
# from sindy_autoencoder_cps.callbacks import SequentialLossWeightsCallback


HPARAMS = dict(
    learning_rate=1e-3,
    input_dim=3, 
    latent_dim=3,
    enc_hidden_sizes=[512, 128, 64, 32],
    dec_hidden_sizes=[33, 64, 128, 512],
    activation='tanh',
    debug=False,
    validdation_split=.1,
    batch_size=32,
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
    sequential_thresholding=False,
    sequential_thresholding_freq = 5,
    sequential_thresholding_thres = 0.05,
    loss_weight_sindy_x=2e2,
    loss_weight_sindy_z=1e1,
    loss_weight_sindy_regularization=5e-5,
    dataset='base'
)

def train():
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        # track_grad_norm='inf',
        gradient_clip_val=.1,
        gpus=gpus,
        max_epochs=10_000,
        deterministic=True,
        precision=64,
        # stochastic_weight_avg=True,
        callbacks=[SequentialThresholdingCallback()
                   # ,SequentialLossWeightsCallback()
                   ])
    model = SINDyAutoencoder(**HPARAMS)
    dm = ThreeTankDataModule(validdation_split=HPARAMS['validdation_split'],
                             batch_size=HPARAMS['batch_size'],
                             dl_num_workers=HPARAMS['dl_num_workers'],
                             debug=HPARAMS['debug'],
                             dataset=HPARAMS['dataset'])
    trainer.fit(model, dm)

def continue_training():
    LAST_CKP = 'lightning_logs/version_46/checkpoints/epoch=2-step=8522.ckpt'
    model = SINDyAutoencoder.load_from_checkpoint(LAST_CKP, **HPARAMS)
    trainer = pl.Trainer(resume_from_checkpoint=LAST_CKP, max_epochs=100000, gpus=1)
    dm = ThreeTankDataModule(validdation_split=HPARAMS['validdation_split'],
                             batch_size=HPARAMS['batch_size'],
                             dl_num_workers=HPARAMS['dl_num_workers'],
                             debug=HPARAMS['debug'],
                             dataset=HPARAMS['dataset'])
    trainer.fit(model, dm)




if __name__ == '__main__':
    # continue_training()
    train()
