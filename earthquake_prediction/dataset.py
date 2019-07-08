import random
from torch.utils.data import Dataset


class SeismicDataSequence(Dataset):
    def __init__(self, df, dataloader_length, input_sequence_length=150_000):
        super(SeismicDataSequence).__init__()
        self.df = df
        self.dataloader_length = dataloader_length
        self.input_sequence_length = input_sequence_length
        self.df_length = len(self.df.index)

    def __getitem__(self, item):
        idx = int(random.uniform(0, self.df_length - self.input_sequence_length))
        sub_df = self.df.iloc[idx: idx + self.input_sequence_length]
        return sub_df['acoustic_data'].values, sub_df['time_to_failure'].values[-1]

    def __len__(self):
        return self.dataloader_length
