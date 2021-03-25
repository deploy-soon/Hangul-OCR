import sys
sys.path.append("../")
import torch
from torchvision import transforms
import unittest
from data import *


class TestData(unittest.TestCase):

    def test_text_to_image(self, verbose=False):
        char = chr(JAMO_OFFSET)

        fontpaths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf",
        ]
        fonts = [ImageFont.truetype(path, 40) for path in fontpaths]

        img = text_to_image(char, width=64, height=64,
                            font=fonts[0], color=255,
                            stroke_width=1)
        img = np.array(img)

        if verbose:
            for i in range(64):
                for j in range(64):
                    print(" " if img[i][j] == 0 else "O", end="")
                print()

    def test_dataset1(self):
        dataset = HangulDataset()

        for img, label in dataset:
            self.assertEqual(sum(label), 3)

    def _test_dataset2(self):
        dataset = HangulDataset()

        for img, label in dataset:
            self.assertEqual(sum(label), 3)

    def test_text_to_label(self):
        char = "가"
        i, j, k = text_to_label(char)
        self.assertEqual(i, 0)
        self.assertEqual(k, 0)
        char = "난"
        i, j, k = text_to_label(char)
        self.assertEqual(i, 2)
        self.assertEqual(j, 0)
        self.assertEqual(k, 3+1) # label 0 is non-tail

    def test_testset(self):
        test_transform = transforms.Compose([
            transforms.Resize((64 - 8, 64 - 8)),
            transforms.ToTensor(),
            transforms.Pad(4, fill=1),
            transforms.Lambda(lambda x: 1-x)
            #transforms.Normalize((0.5, ), (1.0, ))
        ])
        dataset = TestDataset(transform=test_transform)
        for img, label in dataset:
            self.assertEqual(sum(label), 3)
            char = label_to_text(label)
            print(char)
            img = np.array(img.squeeze())
            for i in range(64):
                for j in range(64):
                    print("{:3d}".format(int(img[i][j]*255)), end="")
                    #print(" " if img[0][i][j] == 0 else "O", end="")
                print()
            break


if __name__ == "__main__":
    unittest.main()

