import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn, optim
from LightningGeneralFunctions import general_test_step, general_train_step, general_validation_step, \
    general_test_loader, general_train_loader


LEARNING_RATE = 1e-4
NUM_FEATURES = 21
NUM_CHANNELS = 10
EMG_FOLDER = 'e8l'
GLOVE_FOLDER = 'gv03'
WINDOW_SIZE = 200


class GloveNet(pl.LightningModule):
    def __init__(self):
        super(GloveNet, self).__init__()
        self.lr = LEARNING_RATE
        self.batch_size = 32
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * (WINDOW_SIZE//2) * (NUM_CHANNELS//2), 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, NUM_FEATURES)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        return general_train_step(self, batch)

    def validation_step(self, batch, batch_idx):
        return general_validation_step(self, batch)

    def test_step(self, batch, batch_idx):
        return general_test_step(self, batch)

    def validation_epoch_end(self, outputs):
        loss = torch.stack(outputs).mean()
        print(torch.stack(outputs), torch.stack(outputs).shape)
        self.log('val_loss_mean', loss)

    def test_epoch_end(self, outputs):
        loss = torch.stack(outputs).mean()
        print(torch.stack(outputs), torch.stack(outputs).shape, len(outputs))
        self.log('test_loss_mean', loss)

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
        save_last=True
    )

    trainer = pl.Trainer(gpus=1, auto_lr_find=False, auto_scale_batch_size=False, max_epochs=50,
                         callbacks=[checkpoint_callback]
                         )
    trainer.fit(model=net)
    trainer.test(model=net)
    best = GloveNet.load_from_checkpoint(checkpoint_callback.best_model_path)
    print(checkpoint_callback.best_model_path)
    trainer.test(model=best)

#  tensorboard --logdir ./lightning_logs
