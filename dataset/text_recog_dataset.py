import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import PIL
import random
import numpy as np

from model.utils import make_std_mask


class VLPTextRecognitionDataset(Dataset):

    def __init__(self, data, tokenizer, max_text_len, training=False):
        data_with_text = []

        start_token_id, end_token_id = tokenizer.convert_tokens_to_ids(
            ['[START]', '[END]'])
        for img, text in data:
            # for img, _, text in data:
            tokenized_text = tokenizer.tokenize(text)
            tokenized_text = tokenizer.convert_tokens_to_ids(
                tokenized_text)[:max_text_len - 2]

            tokenized_text = [start_token_id] + tokenized_text + [end_token_id]

            tgt = tokenized_text[:-1]
            tgt_y = tokenized_text[1:]

            padding_len = max_text_len - len(tgt)
            tokenized_text += [0] * padding_len
            tgt += [0] * padding_len
            tgt_y += [0] * padding_len

            tgt = torch.tensor(tgt)
            tgt_y = torch.tensor(tgt_y)

            tgt_mask = make_std_mask(tgt, 0)

            data_with_text.append((img, tgt, tgt_y, tgt_mask))

        self.data = data_with_text
        self.training = training

    def __getitem__(self, index):
        image_path, tgt, tgt_y, tgt_mask = self.data[index]
        # TODO: remove
        image_path = "../notebooks/" + image_path
        image = PIL.Image.open(image_path)

        if self.training and random.random() < 0.5:
            image = PIL.Image.fromarray(np.invert(image))

        image = F.to_tensor(image)

        return image, tgt, tgt_y, tgt_mask

    def __len__(self):
        return len(self.data)
