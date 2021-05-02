import argparse
import numpy as np

import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms

from data import HangulDataset, TestDataset, JAMO1, JAMO2, JAMO3, label_to_text
import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(preds, trues):
    assert preds.shape[0] == trues.shape[0]
    #return sum([(p == t).all() for p, t in zip(preds, trues)])
    return sum([label_to_text(p) == label_to_text(t) for p, t in zip(preds, trues)])

def accuracy_all(preds, trues):
    assert preds.shape[0] == trues.shape[0]
    scores = [0] * 4
    for pred, true in zip(preds, trues):
        jamo1 = np.argmax(pred[:JAMO1])
        jamo2 = np.argmax(pred[JAMO1:JAMO1+JAMO2])
        jamo3 = np.argmax(pred[JAMO1+JAMO2:])
        score = int(true[jamo1] + true[JAMO1+jamo2] + true[JAMO1+JAMO2+jamo3])
        scores[score] += 1
    return scores


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Train:

    def __init__(self, args):
        self.args = args

        transform = transforms.Compose([
            #transforms.Resize(72),
            transforms.RandomAffine(degrees=15, scale=(.9, 1.1), shear=0),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.RandomResizedCrop(64, scale=(0.7, 1.0)),
            transforms.ToTensor(),
            transforms.RandomErasing(scale=(0.02, 0.04), ratio=(0.3, 1.6)),
            transforms.Normalize((0.5, ), (1.0, ))
        ])

        dataset = HangulDataset(transform=transform, image_size=72)
        self.train_num = int(args.train_ratio * len(dataset))
        self.vali_num = len(dataset) - self.train_num
        train_set, vali_set = torch.utils.data.random_split(dataset, [self.train_num, self.vali_num])
        self.train_loader = torch.utils.data.DataLoader(train_set, args.batch_size,
                                                        shuffle=True, num_workers=8)
        self.vali_loader = torch.utils.data.DataLoader(vali_set, args.batch_size,
                                                       shuffle=False, num_workers=8)

        test_transform = transforms.Compose([
            transforms.Resize((args.image_size-24, args.image_size-24)),
            transforms.Pad(12, fill=255),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1 - x),
            transforms.Normalize((0.5, ), (1.0, ))
        ])
        testset = TestDataset(transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(testset, args.test_batch_size,
                                                       shuffle=False, num_workers=8)

        self.model = models.__dict__[args.arch](num_classes=JAMO1+JAMO2+JAMO3, **vars(args)).to(device)
        print("model parameters", count_parameters(self.model))
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

        if args.pretrained:
            self.model.load_state_dict(torch.load(args.checkpoint))

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
            score += accuracy(preds.cpu().numpy(), ys.cpu().numpy())

            vali_loss += loss.item() * xs.size(0)
        #print("validate acc:", '%.3f' % (score / self.vali_num))
        return vali_loss / self.vali_num, score / self.vali_num

    def test(self):
        self.model.eval()
        score = 0
        score_all = [0] * 4

        scores_map = {}
        for xs, ys in tqdm.tqdm(self.test_loader):
            xs = xs.to(device)
            preds = self.model(xs)

            #preds = torch.sigmoid(preds).data > 0.5
            ys = ys.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            score += accuracy(preds, ys)
            for i, val in enumerate(accuracy_all(preds, ys)):
                score_all[i] += val

            for y, pred in zip(ys, preds):
                char_y = label_to_text(y)
                char_p = label_to_text(pred)

                if char_y != char_p:
                    scores_map[char_y] = scores_map.setdefault(char_y, 0) + 1

                    #print(f"true: {char_y}, pred:{char_p}")

        acc = score / len(self.test_loader.dataset)
        print("Test Accuracy: {:.4f}".format(acc))
        print("Zero Acc: {:.4f} One Acc: {:.4f} Two Acc: {:.4f} Thr Acc: {:.4f}".format(
            score_all[0] / sum(score_all), score_all[1] / sum(score_all),
            score_all[2] / sum(score_all), score_all[3] / sum(score_all),
        ))

    def inference(self):
        self.model.eval()
        for xs, ys in self.vali_loader:
            xs = xs.to(device)
            preds = self.model(xs)

            ys = ys.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            for y, pred in zip(ys, preds):
                char_y = label_to_text(y)
                char_p = label_to_text(pred)
                if char_y != char_p:
                    print(f"true: {char_y}, pred:{char_p}")
            break

    def run(self):
        best_loss = 999
        best_acc = 0
        for epoch in range(self.args.epochs):
            train_loss = self.train()
            vali_loss, vali_acc = self.validate()
            print("epochs: {}, train_loss: {:.4f}, vali_loss: {:.4f}, vali_acc: {:.4f}".format(epoch+1, train_loss, vali_loss, vali_acc))
            if best_loss > vali_loss:
                best_loss = vali_loss
                best_acc = vali_acc
                if self.args.checkpoint is not None:
                    torch.save(self.model.state_dict(), self.args.checkpoint)
        self.inference()
        print("best acc", best_acc)


def main():

    parser = argparse.ArgumentParser(description="OCR for Hangul",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--image_size', type=int, default=64)

    #  vit
    parser.add_argument('--depth', type=int, default=8)

    # train
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on testset')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--checkpoint', type=str,
                        help='save just model parameter')

    #optimizer
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    args = parser.parse_args()

    train = Train(args)
    if args.evaluate:
        train.test()
    else:
        train.run()

if __name__ == "__main__":
    main()
