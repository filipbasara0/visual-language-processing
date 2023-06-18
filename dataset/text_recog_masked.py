import torch
import torchvision.transforms.functional as F
from nltk import word_tokenize
import textwrap
from PIL import Image, ImageFont, ImageDraw
import random
import numpy as np
import os

import time
from multiprocessing import Pool, Process, Queue, Manager
from queue import Empty

random.seed(321564654)
np.random.seed(321564654)

CHUNK_SIZE = 128

data = None

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
    if font_path in invalid_fonts or any(i_f in font_path.lower()
                                         for i_f in invalid_font_families):
        continue
    processed_fonts.append(font_path)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return subsequent_mask == 0


def make_std_mask(tgt, pad):
    tgt_mask = np.expand_dims(tgt != pad, axis=-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.shape[-1])
    return tgt_mask


ALIGNMENTS = ["center", "left", "right", "down", "up"]
font_category_distributions = {
    "small": (30, 30, 55),
    "medium": (23, 30, 65),
    "large": (20, 30, 75)
}


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

    x_text, y_text = min(5, x_text), min(5, y_text)
    return x_text, y_text


def create_image(text,
                 filename,
                 image_size=512,
                 font="arial.ttf",
                 font_size=20,
                 bg_color=(255, 255, 255),
                 bg='white',
                 align="left",
                 width=636,
                 height=636,
                 text_width=40,
                 resize=True,
                 save_image=True):
    font_size = int(font_size / 1.5)

    # text width for wrapping
    text = '\n'.join(textwrap.wrap(text, width=text_width))  #, width=18
    font = ImageFont.truetype(font, font_size)

    image = Image.new(mode="RGB", size=(width, height),
                      color="white")  # (700, 620)
    draw = ImageDraw.Draw(image)

    l, t, r, b = draw.multiline_textbbox((0, 0), text, font=font)
    l_offset = abs(l) + 10
    t_offset = abs(t) + 10
    l = l_offset
    t = t_offset
    r += l_offset
    b += t_offset
    draw.text((l, t), text, font=font, fill="black")
    image = image.crop((0, 0, r + 10, b + 10))

    if resize:
        image = image.resize((image_size, image_size),
                             Image.Resampling.LANCZOS)
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
    font_size, min_text_width, max_text_width = font_category_distributions[
        text_category]
    return font_size, random.randrange(min_text_width, max_text_width)


def generate_unmasked_image(text, save_images=False):
    num_tokens = len(word_tokenize(text))
    text_category = get_text_category(num_tokens)

    font_path = sample_random_font()
    font_size, text_width = sample_fs_and_tw(text_category)
    alignment = sample_random_alignment()

    return create_image(
        text,
        font=font_path,
        filename=None,
        bg_color=(255, 255, 255),
        bg='white',
        align=alignment,
        # height=800,
        # width=800,
        font_size=font_size,
        text_width=text_width,
        resize=True,
        save_image=save_images)


def prepare_data(data, tokenizer, max_text_len):
    start_token_id, end_token_id = tokenizer.convert_tokens_to_ids(
        ['[START]', '[END]'])
    data_pairs = []
    idx = 0
    # for text in data[:100000]:
    for text in data:
        if idx % 100000 == 0:
            print(idx)
        idx += 1
        text = text.replace("@-@",
                            "-").replace("@.@",
                                         ".").replace("@,@",
                                                      ",").replace("@;@", ";")
        tokenized_text = tokenizer.tokenize(text)
        tokenized_text = tokenizer.convert_tokens_to_ids(
            tokenized_text)[:max_text_len - 2]

        tokenized_text = [start_token_id] + tokenized_text + [end_token_id]

        data_pairs.append((text, tokenized_text))

    return data_pairs


def generate_data_instance(text, tokenized_text, max_text_len, training=True):
    image = generate_unmasked_image(text, save_images=False)

    if training and random.random() < 0.5:
        image = Image.fromarray(np.invert(image))

    tgt = tokenized_text[:-1]
    tgt_y = tokenized_text[1:]

    padding_len = max_text_len - len(tgt)
    tokenized_text += [0] * padding_len
    tgt += [0] * padding_len
    tgt_y += [0] * padding_len

    tgt = np.array(tgt)
    tgt_y = np.array(tgt_y)

    tgt_mask = make_std_mask(tgt, 0)

    return image, tgt, tgt_y, tgt_mask


def mask_tokens(tokens, max_text_len, mask_prob=0.15):
    # tokens = word_tokenize(text)[:max_text_len]
    tokens_range = range(len(tokens))

    n_sample = round(len(tokens_range) * mask_prob)
    mask_tokens_idxs = random.sample(tokens_range, n_sample)

    return [
        t if idx not in mask_tokens_idxs else _tokenizer.mask_token_id
        for idx, t in enumerate(tokens)
    ], mask_tokens_idxs


def gen_process(text, tokenized_text, max_text_len, training):
    masked_tokens, mask_tokens_idxs = mask_tokens(
        tokenized_text[1:len(tokenized_text) - 1], max_text_len)
    mask_tokens_idxs += [-1] * (max_text_len - len(mask_tokens_idxs))
    masked_text = _tokenizer.decode(masked_tokens).replace(" # # ", "##")
    image, tgt, tgt_y, tgt_mask = generate_data_instance(
        masked_text, tokenized_text, max_text_len, training)
    return (image, tgt, tgt_y, tgt_mask, mask_tokens_idxs)


def divide_chunks(l, n):
    split = []
    for i in range(0, len(l), n):
        split.append(l[i:i + n])
    return split


def collate_batch(batch):
    images = torch.stack([F.to_tensor(b[0]) for b in batch])
    tgt = torch.tensor([b[1] for b in batch])
    tgt_y = torch.tensor([b[2] for b in batch])
    tgt_mask = torch.tensor([b[3] for b in batch])
    mask_tokens_idxs = torch.tensor([b[4] for b in batch])
    return images, tgt, tgt_y, tgt_mask, mask_tokens_idxs


def producer(queue, batch_size, max_text_len, training):
    data_range = list(range(len(data)))
    idx = 0
    while idx < len(data_range):
        if queue.qsize() >= 300:
            continue
        pool = Pool(processes=3)
        chunks = []
        tasks = []
        for index in data_range[idx:idx + CHUNK_SIZE]:
            try:
                text, tokenized_text = data[index]

                def handle_error(error):
                    print("handle_error", error, flush=True)
                    print("text", text, flush=True)
                    print("tokenized_text", tokenized_text, flush=True)
                    print("queue.qsize()", queue.qsize(), flush=True)

                out = pool.apply_async(
                    gen_process,
                    [text, tokenized_text, max_text_len, training],
                    error_callback=handle_error)
                tasks.append(out)
            except Exception as e:
                print("EXCEPTION FOR", e, flush=True)
                print("text", text, flush=True)
                print("tokenized_text", tokenized_text, flush=True)
                print("queue.qsize()", queue.qsize(), flush=True)

        for task in tasks:
            try:
                chunks.append(task.get())
            except Exception as e:
                print("EXCEPTION OUT", e, flush=True)
                print("queue.qsize()", queue.qsize(), flush=True)

        for batch in divide_chunks(chunks, batch_size):
            try:
                queue.put(batch)
            except Exception as e:
                print("EXCEPTION DIVIDE CHUNKS", e, flush=True)
                print("queue.qsize()", queue.qsize(), flush=True)
        # print("producer", idx, queue.qsize())
        pool.close()
        # pool.join()
        idx += CHUNK_SIZE

    queue.put(None)
    print('Producer: Done', flush=True)


def generate(queue):
    i = 0
    while True:
        try:
            batch = queue.get(block=False)
            # batch = queue.get(block=True, )
        except Empty:
            # print('Consumer: got nothing, waiting a while...', flush=True)
            # print("queue size", queue.qsize(), flush=True)
            time.sleep(0.5)
            continue
        if batch is None:
            break
        # print(f'consumer {i}', flush=True)
        i += 1
        yield collate_batch(batch)
    print('Consumer: Done', flush=True)


def start_processes(batch_size, max_text_len, training):
    queue = Queue()
    # queue = Manager().Queue()
    producer_process = Process(target=producer,
                               args=(queue, batch_size, max_text_len,
                                     training))
    producer_process.start()
    return queue


def get_online_generator(train_data, tokenizer, max_text_len, batch_size,
                         training):
    global _tokenizer
    _tokenizer = tokenizer
    global data
    data = prepare_data(train_data, tokenizer, max_text_len)
    random.shuffle(data)
    queue = start_processes(batch_size, max_text_len, training)
    return generate(queue)
