from examples.lorenz.data_module import LorenzDataModule
from sindy_autoencoder_cps.lightning_module import SINDyAutoencoder
import torch
import pytorch_lightning as pl
from sindy_autoencoder_cps.callbacks import SequentialThresholdingCallback


HPARAMS = dict(
    learning_rate=1e-4,
    # adam_eps=1e-2,
    input_dim=3, 
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
    loss_weight_sindy_x=0,
    loss_weight_sindy_z=0,
    loss_weight_sindy_regularization=1e-8,
    # dataset='base'
)

def train():
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        # track_grad_norm='inf',
        # gradient_clip_val=10,
        # deterministic=True,
        precision=64,
        gpus=gpus,
        max_epochs=10_000,
        callbacks=[SequentialThresholdingCallback()
                   # ,SequentialLossWeightsCallback()
                   ])
    model = SINDyAutoencoder(**HPARAMS)
    dm = LorenzDataModule(validdation_split=HPARAMS['validdation_split'],
                             batch_size=HPARAMS['batch_size'],
                             dl_num_workers=HPARAMS['dl_num_workers'],
                             debug=HPARAMS['debug'],
                             # dataset=HPARAMS['dataset']
                          )
    trainer.fit(model, dm)


if __name__ == '__main__':
    # continue_training()
    train()
