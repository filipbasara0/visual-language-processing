import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from nltk import word_tokenize
import textwrap
from PIL import Image, ImageFont, ImageDraw
import random
import numpy as np

from model.utils import make_std_mask

random.seed(42)
np.random.seed(42)

ALIGNMENTS = ["center", "left", "right", "down", "up"]
font_category_distributions = {
    "small": [(56, 22), (50, 22), (58, 22), (50, 26), (46, 28), (40, 32),
              (60, 22), (62, 22)],
    "medium": [(40, 32), (42, 28)],
    "large": [(34, 38)]
}

with open("./fonts/ubuntu_fonts.txt", "r") as f:
    fonts_lines = f.readlines()

invalid_font_families = [
    "kacst",
    "lohit",
    "samyak",
]

invalid_fonts = [
    '/usr/share/fonts/type1/urw-base35/D050000L.t1',
    '/usr/share/fonts/opentype/urw-base35/D050000L.otf',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    '/usr/share/fonts/truetype/Gubbi/Gubbi.ttf',
    '/usr/share/fonts/truetype/fonts-kalapi/Kalapi.ttf',
    '/usr/share/fonts/truetype/sinhala/lklug.ttf',
    '/usr/share/fonts/truetype/Navilu/Navilu.ttf',
    '/usr/share/fonts/truetype/openoffice/opens___.ttf',
    '/usr/share/fonts/truetype/fonts-gujr-extra/padmaa-Medium-0.5.ttf',
    '/usr/share/fonts/truetype/fonts-telu-extra/Pothana2000.ttf',
    '/usr/share/fonts/truetype/malayalam/RaghuMalayalamSans-Regular.ttf',
    '/usr/share/fonts/truetype/fonts-guru-extra/Saab.ttf',
    '/usr/share/fonts/type1/urw-base35/StandardSymbolsPS.t1',
    '/usr/share/fonts/opentype/urw-base35/StandardSymbolsPS.otf',
    '/usr/share/fonts/truetype/fonts-orya-extra/utkal.ttf',
    '/usr/share/fonts/truetype/fonts-telu-extra/vemana2000.ttf', '/usrà¦¤à¦¿',
    '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf'
]

processed_fonts = []
for f_l in fonts_lines:
    font_path = f_l.split(":")[0]
    if font_path in invalid_fonts or any(
            i_f in font_path.lower() for i_f in invalid_font_families):
        continue
    processed_fonts.append(font_path)


def get_text_coords(im_dims, text_dims, align):
    im_width, im_height = im_dims
    text_width, text_height = text_dims
    if align == "center":
        x_text = (im_width - text_width) / 2
        y_text = (im_height - text_height) / 2
    elif align == "right":
        x_text = im_width - text_width - 10
        y_text = 10
    elif align == "down":
        x_text = (im_width - text_width) / 2
        y_text = im_height - text_height - 10
    elif align == "up":
        x_text = (im_width - text_width) / 2
        y_text = 10
    # align left
    else:
        x_text = 10
        y_text = 10
    return x_text, y_text


def create_image(text,
                 filename,
                 font="arial.ttf",
                 size=20,
                 bg_color=(255, 255, 255),
                 bg='white',
                 align="left",
                 width=800,
                 height=800,
                 text_width=40,
                 resize=True,
                 save_image=True):
    # text width for wrapping
    text = '\n'.join(textwrap.wrap(text, width=text_width))  #, width=18
    font = ImageFont.truetype(font, size)

    image = Image.new(mode="RGB", size=(width, height),
                      color=(0, 0, 0))  # (700, 620)
    draw = ImageDraw.Draw(image)

    # text width on image
    text_width, text_height = draw.textsize(text, font=font)
    x_text, y_text = get_text_coords((width, height), (text_width, text_height),
                                     align)

    draw.text((x_text, y_text), text, font=font, fill=(255, 255, 255))
    if resize:
        image = image.resize((384, 384), Image.Resampling.LANCZOS)
    if save_image:
        image.save(filename)
    return image


def get_text_category(num_tokens):
    if num_tokens <= 50:
        return "small"
    elif num_tokens <= 100:
        return "medium"
    else:
        return "large"


def sample_random_font():
    return random.choice(processed_fonts)


def sample_random_alignment():
    return random.choices(ALIGNMENTS, weights=[0.25, 0.2, 0.1, 0.2, 0.25])[0]


def sample_fs_and_tw(text_category):
    category_distributions = font_category_distributions[text_category]
    return random.choice(category_distributions)


def generate_unmasked_image(text, save_images=False):
    num_tokens = len(word_tokenize(text))
    text_category = get_text_category(num_tokens)

    font_path = sample_random_font()
    font_size, text_width = sample_fs_and_tw(text_category)
    alignment = sample_random_alignment()

    return create_image(text,
                        font=font_path,
                        filename=None,
                        bg_color=(255, 255, 255),
                        bg='white',
                        align=alignment,
                        height=800,
                        width=800,
                        size=font_size,
                        text_width=text_width,
                        resize=True,
                        save_image=save_images)


class VLPDatasetOnline(Dataset):

    def __init__(self, data, tokenizer, max_text_len, training=False):
        start_token_id, end_token_id = tokenizer.convert_tokens_to_ids(
            ['[START]', '[END]'])
        data_pairs = []
        seqs = []
        idx = 0
        for text in data:
            if idx % 100000 == 0:
                print(idx)
            idx += 1
            tokenized_text = tokenizer.tokenize(text)
            tokenized_text = tokenizer.convert_tokens_to_ids(
                tokenized_text)[:max_text_len - 2]

            tokenized_text = [start_token_id] + tokenized_text + [end_token_id]

            data_pairs.append((tokenized_text, text))

        self.data = data_pairs
        self.max_text_len = max_text_len
        self.training = training

    def __getitem__(self, index):
        tokenized_text, text = self.data[index]

        image = generate_unmasked_image(text, save_images=False)

        if self.training and random.random() < 0.5:
            image = Image.fromarray(np.invert(image))

        image = F.to_tensor(image)

        tgt = tokenized_text[:-1]
        tgt_y = tokenized_text[1:]

        padding_len = self.max_text_len - len(tgt)
        tokenized_text += [0] * padding_len
        tgt += [0] * padding_len
        tgt_y += [0] * padding_len

        tgt = torch.tensor(tgt)
        tgt_y = torch.tensor(tgt_y)

        tgt_mask = make_std_mask(tgt, 0)

        return image, tgt, tgt_y, tgt_mask

    def __len__(self):
        return len(self.data)
