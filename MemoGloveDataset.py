import torch
from torch.utils.data import Dataset


class GloveDataset(Dataset):
    def __init__(self, emg_folder, glove_folder, memo_folder, window_size, fold_number, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.glove_path = '../own_data/' + glove_folder + '/' + str(window_size) + '/'
        self.emg_path = '../own_data/' + emg_folder + '/' + str(window_size) + '/'
        self.memo_path = '../own_data/' + memo_folder + '/' + str(window_size) + '/'

        self.emg_data = \
            torch.load(self.emg_path + str(fold_number) + '.pt')
        self.glove_data = \
            torch.load(self.glove_path + str(fold_number) + '.pt')
        self.previous = \
            torch.load(self.memo_path + str(fold_number) + '.pt')

    def __len__(self):
        return len(self.glove_data)

    def __getitem__(self, idx):
        emg = self.emg_data[idx]
        glove = self.glove_data[idx]
        previous = self.previous[idx]
        if self.transform:
            emg = self.transform(emg)
        if self.target_transform:
            glove = self.target_transform(glove)
            previous = self.target_transform(previous)

        return emg, glove, previous

    def expand(self, fold_number):
        new_emg = torch.load(self.emg_path + str(fold_number) + '.pt')
        new_glove = torch.load(self.glove_path + str(fold_number) + '.pt')
        new_previous = torch.load(self.memo_path + str(fold_number) + '.pt')

        self.emg_data = torch.vstack((self.emg_data, new_emg))
        self.glove_data = torch.vstack((self.glove_data, new_glove))
        self.previous = torch.vstack((self.previous, new_previous))
