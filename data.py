import os
import glob
import random
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from torch.utils.data import Dataset

JAMO_OFFSET = 0xAC00
JAMO1 = 19
JAMO2 = 21
JAMO3 = 28


def label_to_text(label):
    jamo1 = label[:JAMO1]
    jamo2 = label[JAMO1:JAMO1+JAMO2]
    jamo3 = label[JAMO1+JAMO2:]
    i = np.argmax(jamo1)
    j = np.argmax(jamo2)
    k = np.argmax(jamo3)
    char = chr(i * 588 + j * 28 + k + JAMO_OFFSET)
    return char

def text_to_label(char):
    idx = ord(char) - JAMO_OFFSET
    jamo1 = idx // (JAMO2 * JAMO3)
    jamo3 = idx % JAMO3
    jamo2 = ((idx - jamo3) % (JAMO2 * JAMO3)) // 28
    return jamo1, jamo2, jamo3

def label_to_one_hot(i, j, k):
    label = np.zeros((JAMO1 + JAMO2 + JAMO3,), np.float32)
    label[i] = 1
    label[JAMO1 + j] = 1
    label[JAMO1 + JAMO2 + k] = 1
    return label

def text_to_image(char, width, height,
                  font,
                  color,
                  stroke_width,
                  start_point=(10, 6)):
    img = np.zeros((height, width), np.uint8)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(start_point, char, font=font, fill=color,
              stroke_width=stroke_width)

    return img_pil

class HangulDataset(Dataset):

    def __init__(self, transform=None,
                 image_size=72, font_size=45):
        self.transform = transform

        chars = []
        for i in range(JAMO1):
            for j in range(JAMO2):
                for k in range(JAMO3):
                    char = chr(i * 588 + j * 28 + k + JAMO_OFFSET)
                    chars.append((char, i, j, k))
        self.chars = chars
        self.image_size = image_size

        fontpaths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf",
        ]
        self.fonts = [ImageFont.truetype(path, font_size) for path in fontpaths]

    def __len__(self):
        return len(self.chars)

    def __getitem__(self, idx):
        char, i, j, k = self.chars[idx]
        img = text_to_image(char, width=self.image_size,
                            height=self.image_size,
                            font=random.choice(self.fonts),
                            color=random.randint(200, 255),
                            stroke_width=random.randint(1, 2))
        label = label_to_one_hot(i, j, k)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TestDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        data_path = "/home/soon/chardata/*/*/*/*.png"
        file_list = glob.glob(data_path)

        images = []
        for path in file_list[:200000]:
            _path = os.path.normpath(path).split(os.path.sep)
            if len(_path) > 3 and _path[-3].startswith("Hangul"):
                char = _path[-2]
                i, j, k = text_to_label(char)
                label = label_to_one_hot(i, j, k)
                images.append((label, path))
        self.images = images
        print("load test images:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label, path = self.images[idx]
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    data = TestDataset()




