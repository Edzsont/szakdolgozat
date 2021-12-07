import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn, optim
from LightningGeneralFunctions import general_train_step, general_test_step, general_validation_step, \
    general_train_loader, general_test_loader

LEARNING_RATE = 1e-4
NUM_FEATURES = 21
NUM_CHANNELS = 12
GLOVE_FOLDER = 'gv03'
EMG_FOLDER = 'e12'
WINDOW_SIZE = 200


class GloveNet(pl.LightningModule):
    def __init__(self):
        super(GloveNet, self).__init__()
        self.lr = LEARNING_RATE
        self.batch_size = 32

        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(WINDOW_SIZE * NUM_CHANNELS, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, NUM_FEATURES)
        )

    def forward(self, x):
        x = self.regression(x)
        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        return general_train_step(self, batch)

    def validation_step(self, batch, batch_idx):
        return general_validation_step(self, batch)

    def test_step(self, batch, batch_idx):
        return general_test_step(self, batch)

    def train_dataloader(self):
        return general_train_loader(EMG_FOLDER, GLOVE_FOLDER, WINDOW_SIZE, self.batch_size)

    def test_dataloader(self):
        return general_test_loader(EMG_FOLDER, GLOVE_FOLDER, WINDOW_SIZE, self.batch_size, 0)

    def val_dataloader(self):
        return general_test_loader(EMG_FOLDER, GLOVE_FOLDER, WINDOW_SIZE, self.batch_size, 4)


if __name__ == '__main__':
    print('CUDA is available: ', torch.cuda.is_available())

    net = GloveNet()

    test_net = GloveNet()
    test = torch.randn((1, 1, WINDOW_SIZE, NUM_CHANNELS))
    res = test_net(test)
    print(res.shape)

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        save_top_k=3,
        mode='min',
        # period=5,
        save_last=True
    )

    trainer = pl.Trainer(gpus=1, auto_lr_find=False, auto_scale_batch_size=False, max_epochs=250,
                         callbacks=[checkpoint_callback]
                         )

    trainer.fit(model=net)
    trainer.test(model=net)
    best = GloveNet.load_from_checkpoint(checkpoint_callback.best_model_path)
    print(checkpoint_callback.best_model_path)
    trainer.test(model=best)

#  tensorboard --logdir ./lightning_logs
