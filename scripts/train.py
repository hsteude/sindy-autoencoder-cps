from three_tank_data.data_module import ThreeTankDataModule
from sindy_autoencoder_cps.lightning_module import SINDyAutoencoder
import torch
import pytorch_lightning as pl


HPARAMS = dict(
    learning_rate=.0005,
    network_hidden_size=100,
    input_dim=10000, 
    latent_dim=3,
    number_candidate_functions=10,
    validdation_split=.1,
    batch_size=128,
    dl_num_workers=24,
    max_epochs=10_000
)

def train():
        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=10_000)
        model = SINDyAutoencoder(**HPARAMS)
        dm = ThreeTankDataModule(validdation_split=HPARAMS['validdation_split'],
                                 batch_size=HPARAMS['batch_size'],
                                 dl_num_workers=HPARAMS['dl_num_workers'])
        trainer.fit(model, dm)



if __name__ == '__main__':
    # continue_training()
    train()
