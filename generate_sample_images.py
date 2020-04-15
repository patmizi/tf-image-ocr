import os
from PIL import Image

from pandas import np

from lib.utils.image_utils import paint_text
from lib.constants import ALPHABET

OUT_PATH = './test/generated'
img_w = 512
img_h = 64


def generate_images(batch_size):
    for i in range(batch_size):
        word_len = np.random.randint(3, 25 + 1)  # max_string_len // 2
        word = ""

        for k in range(word_len):
            word += ALPHABET[np.random.randint(0, len(ALPHABET) - 1)]

        save_path = os.path.join(OUT_PATH, "image-{}.png".format(i))
        img_array = paint_text(word, img_w, img_h, True, True, True)
        img = Image.fromarray(img_array).convert('RGB')
        img.save(save_path)


if __name__ == '__main__':
    generate_images(2)
