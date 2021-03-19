import argparse

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms

from data import HangulDataset, JAMO1, JAMO2, JAMO3
from models import resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(preds, trues):
    assert preds.shape[0] == trues.shape[0]
    return sum([(p == t).all() for p, t in zip(preds, trues)])

class Train:

    def __init__(self, args):
        self.args = args

        transform = transforms.Compose([
            transforms.RandomRotation(2.8),
            transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(scale=(0.02, 0.16), ratio=(0.3, 1.6)),
            transforms.Normalize((0.5, ), (1.0, ))
        ])

        dataset = HangulDataset(transform=transform)
        self.train_num = int(args.train_ratio * len(dataset))
        self.vali_num = len(dataset) - self.train_num
        train_set, vali_set = torch.utils.data.random_split(dataset, [self.train_num, self.vali_num])
        self.train_loader = torch.utils.data.DataLoader(train_set, args.batch_size,
                                                        shuffle=True, num_workers=8)
        self.vali_loader = torch.utils.data.DataLoader(vali_set, args.batch_size,
                                                       shuffle=False, num_workers=8)

        self.model = resnet50(num_classes=JAMO1+JAMO2+JAMO3).to(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

    def train(self):
        self.model.train()
        train_loss = 0.0
        for xs, ys in self.train_loader:
            xs, ys = xs.to(device), ys.to(device)
            preds = self.model(xs)
            loss = self.criterion(preds, ys.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * xs.size(0)
        return train_loss / self.train_num

    def validate(self):
        self.model.eval()
        vali_loss = 0.0
        score = 0

        for xs, ys in self.vali_loader:
            xs, ys = xs.to(device), ys.to(device)
            preds = self.model(xs)
            loss = self.criterion(preds, ys.float())

            preds = torch.sigmoid(preds).data > 0.5
            score += accuracy(ys.cpu().numpy(), preds.cpu().numpy())

            vali_loss += loss.item() * xs.size(0)
        print("validate acc:", '%.3f' % (score / self.vali_num))
        return vali_loss / self.vali_num

    def run(self):
        for epoch in range(self.args.epochs):
            train_loss = self.train()
            vali_loss = self.validate()
            print("epochs: {}, train_loss: {:.4f}, vali_loss: {:.4f}".format(epoch+1, train_loss, vali_loss))


def main():

    parser = argparse.ArgumentParser(description="OCR for Hangul",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model

    # train
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_ratio', type=float, default=0.70)
    #optimizer
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()

    train = Train(args)
    train.run()

if __name__ == "__main__":
    main()
