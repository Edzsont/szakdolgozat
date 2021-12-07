import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn, optim
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from LightningGeneralFunctions import preprocess, loss_function, memo_test_step, memo_time_test
from Losses import my_loss
from Normalization import feature_scale, inv_feature_standardize

from MemoGloveDataset import GloveDataset

LEARNING_RATE = 1e-4
NUM_FEATURES = 21
NUM_CHANNELS = 10
if NUM_CHANNELS == 12:
    EMG_FOLDER = 'e12'
else:
    EMG_FOLDER = 'e8l'

if NUM_FEATURES == 11:
    GLOVE_FOLDER = 'gv01'
elif NUM_FEATURES == 13:
    GLOVE_FOLDER = 'gv02'
else:
    GLOVE_FOLDER = 'gv03'

if GLOVE_FOLDER == 'gv01':
    PREV_FOLDER = 'pv01'
elif GLOVE_FOLDER == 'gv02':
    PREV_FOLDER = 'pv02'
else:
    PREV_FOLDER = 'pv03'

WINDOW_SIZE = 64


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
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * (WINDOW_SIZE//2) * (NUM_CHANNELS//2) + NUM_FEATURES, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, NUM_FEATURES)
        )

    def forward(self, x, z):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = torch.hstack((x, z))
        x = self.classifier(x.float())
        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        x, y, z = preprocess(x, y, z)

        logits = self.forward(x, z)
        loss = loss_function(logits.float(), y.float())
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        x, y, z = preprocess(x, y, z)

        logits = self.forward(x, z)
        loss = loss_function(logits, y)
        self.log('validation_loss', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        return memo_test_step(self, batch)
        # return memo_time_test(self, batch)
    '''
    def test_epoch_end(self, outputs):
        loss = torch.stack(outputs).mean()
        print(torch.stack(outputs))
        self.log('exec_time_mean', loss)
    '''
    def train_dataloader(self):
        train_data = GloveDataset(emg_folder=EMG_FOLDER, glove_folder=GLOVE_FOLDER, memo_folder=PREV_FOLDER,
                                  fold_number=1, window_size=WINDOW_SIZE)
        train_data.expand(fold_number=2)
        train_data.expand(fold_number=3)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

        return train_loader

    def test_dataloader(self):
        test_data = GloveDataset(emg_folder=EMG_FOLDER, glove_folder=GLOVE_FOLDER, memo_folder=PREV_FOLDER,
                                 fold_number=0, window_size=WINDOW_SIZE)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, drop_last=True)

        return test_loader

    def val_dataloader(self):
        validation_data = GloveDataset(emg_folder=EMG_FOLDER, glove_folder=GLOVE_FOLDER, memo_folder=PREV_FOLDER,
                                       fold_number=4, window_size=WINDOW_SIZE)
        validation_loader = DataLoader(validation_data, batch_size=self.batch_size, shuffle=False, drop_last=True)

        return validation_loader


if __name__ == '__main__':
    print('CUDA is available: ', torch.cuda.is_available())

    net = GloveNet()

    test_net = GloveNet()
    test = torch.randn((1, 1, WINDOW_SIZE, NUM_CHANNELS))
    test_z = torch.randn((1, NUM_FEATURES))
    res = test_net(test, test_z)
    print(res.shape)

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        save_top_k=3,
        mode='min',
        save_last=True
    )

    early_stop_callback = EarlyStopping(monitor='validation_loss', min_delta=0.1, patience=5, verbose=False)
    trainer = pl.Trainer(gpus=1, auto_lr_find=False, auto_scale_batch_size=False, max_epochs=50,
                         callbacks=[checkpoint_callback]
                         )

    trainer.fit(model=net)
    trainer.test(model=net)
    best = GloveNet.load_from_checkpoint(checkpoint_callback.best_model_path)
    print(checkpoint_callback.best_model_path)
    trainer.test(model=best)


#  tensorboard --logdir ./lightning_logs
