import argparse
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

from earthquake_prediction import utils
from earthquake_prediction.dataset import SeismicDataSequence
from earthquake_prediction.net_modules import MaxAbsolutePooling, AdaptiveMaxAbsolutePooling

parser = argparse.ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--num_epochs')


class HighLevelCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp_size = 8096
        self.reduction = AdaptiveMaxAbsolutePooling(self.inp_size)

        cnn_layers = []
        prev_filter = 1
        for filter_size, stride in [(8, 2), (8, 1), (16, 2), (16, 1), (32, 2), (32, 1), (64, 2), (64, 1), (128, 2), (128, 1)]:
            cnn_layers.extend(
                [
                    nn.BatchNorm1d(prev_filter),
                    nn.Conv1d(prev_filter, filter_size, kernel_size=3, stride=stride),
                    nn.ReLU(),
                    # MaxAbsolutePooling(pool_size=4, stride=stride)
                ]
            )
            prev_filter = filter_size

        self.cnn = nn.Sequential(*cnn_layers)

    def forward(self, inp):
        x = self.reduction(inp)
        x = self.cnn(x)
        return x


class TimeSeriesLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 128, batch_first=True)

    def forward(self, inp):
        out, _ = self.lstm(inp)
        return out[:, -1]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HighLevelCNNEncoder()
        self.lstm = TimeSeriesLSTM()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, inp):
        x = self.encoder(inp)
        b, c, s = x.shape
        x = self.lstm(x.permute(0, 2, 1))

        x = self.fc(x.view(b, -1))
        return x


class Experiment:
    def __init__(self, name, df, batch_size=16, device=torch.device('cuda')):
        self.name = name
        self.df = df
        self.batch_size = batch_size
        self.device = device
        # value where the following time to failure is higher -> earthquake happened
        self.train_val_split_idx = 585568143 + 1
        self.logs_path = Path('./logs')
        self.models_path = Path('./models')
        self.net = Net()

    def train(self, num_epochs=100):
        train_dataset = SeismicDataSequence(
            self.df.iloc[:self.train_val_split_idx],
            self.batch_size
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

        val_dataset = SeismicDataSequence(
            self.df.iloc[self.train_val_split_idx:],
            self.batch_size
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

        net = self.net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=.5)
        criterion = nn.L2Loss()
        stats_tracker = utils.StatsTracker()

        for i in range(num_epochs):
            scheduler.step()
            epoch_logger = utils.EpochLogger()
            with tqdm(total=len(train_loader)) as pbar:
                net.train()
                for inp, target in train_loader:
                    inp = inp.float().unsqueeze(1).to(self.device)
                    target = target.float().to(self.device)
                    pred = net(inp).squeeze()

                    loss = criterion(pred, target)
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_logger.update({'train_loss': loss.item()})
                    pbar.set_description('Train epoch: {}, MAE: {:.2f}'.format(i, epoch_logger.metrics['train_loss']))
                    pbar.update()

            with tqdm(total=len(val_loader)) as pbar:
                with torch.no_grad():
                    for inp, target in val_loader:
                        inp = inp.float().unsqueeze(1).to(self.device)
                        target = target.float().to(self.device)
                        pred = net(inp).squeeze()
                        loss = criterion(pred, target)

                        epoch_logger.update({'val_loss': loss.item()})
                        pbar.set_description('Val epoch: {}, MAE: {:.2f}'.format(i, epoch_logger.metrics['val_loss']))
                        pbar.update()

            stats_tracker.add_entry(epoch_logger.metrics)
            if stats_tracker.compare('val_loss'):
                tqdm.write('Saving model...')
                self.save()

        stats_tracker.save(self.logs_path / self.name)

    def predict(self, inp):
        with torch.no_grad():
            self.net.eval()
            return self.net(inp).item()

    def load(self):
        self.net.load_state_dict(torch.load(self.models_path / self.name))

    def save(self):
        torch.save(self.net.state_dict(), self.models_path / self.name)


if __name__ == '__main__':
    args = parser.parse_args()

    df = utils.read_train_from_disk()
    exp = Experiment(args.name, df)

    exp.train(int(args.num_epochs))
