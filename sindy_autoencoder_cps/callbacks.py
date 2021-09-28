from pytorch_lightning.callbacks import Callback

class SequentialThresholdingCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        its_time = trainer.current_epoch % pl_module.sequential_thresholding_freq == 0
        its_not_epoch_1 = trainer.current_epoch > 0
        if its_time and its_not_epoch_1:
            print(f'mask before: {pl_module.XI_coefficient_mask}')
            print(f'threshold: {pl_module.sequential_thresholding_thres}')
            pl_module.XI_coefficient_mask = pl_module.XI > pl_module.sequential_thresholding_thres 
            print(f'mask after: {pl_module.XI_coefficient_mask}')
            print(f'coefficients: {pl_module.XI_coefficient_mask}')
            print(pl_module.XI)

