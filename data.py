from PIL import ImageFont, ImageDraw, Image
import numpy as np
from torch.utils.data import Dataset

JAMO_OFFSET = 0xAC00
JAMO1 = 19
JAMO2 = 21
JAMO3 = 28



def text_to_image(char, width, height,
                  font,
                  color,
                  stroke_width):
    img = np.zeros((height, width), np.uint8)
    font = ImageFont.truetype(fontpath, 20)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 6), char, font=font, fill=color, stroke_width=stroke_width)

    return img_pil

class HangulDataset(Dataset):

    def __init__(self):
        chars = []
        for i in range(JAMO1):
            for j in range(JAMO2):
                for k in range(JAMO3):
                    char = chr(i * 588 + j * 28 + k + JAMO_OFFSET)
                    chars.append((char, i, j, k))
        print(chars[0], chars[-1])
        self.chars = chars

        fontpaths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf",
        ]
        self.fonts = [ImageFont.truetype(path, 40) for path in fontpaths]

    def __len__(self):
        return len(self.chars)

    def __getitem__(self, idx):
        char, i, j, k = self.chars[idx]
        img = text_to_image(char, 64, 64,
                            font=random.choice(self.fonts),
                            color=random.randint(200, 255),
                            strock_width=random.randint(1, 2))
        label = np.zeros((JAMO1 + JAMO2 + JAMO3,), np.float32)
        label[i] = 1
        label[JAMO1 + j] = 1
        label[JAMO1 + JAMO2 + k] = 1
        return img, label


