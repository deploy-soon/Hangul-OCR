import sys
sys.path.append("../")
import unittest
from data import *


class TestDataset(unittest.TestCase):

    def test_text_to_image(self):
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
        for i in range(64):
            for j in range(64):
                print(" " if img[i][j] == 0 else "O", end="")
            print()

    def test_dataset(self):
        dataset = HangulDataset()

        for img, label in dataset:
            self.assertEqual(sum(label), 3)




if __name__ == "__main__":
    unittest.main()

