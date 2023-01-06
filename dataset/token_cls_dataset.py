import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import PIL
import random
import numpy as np

from model.utils import make_std_mask

NER_TAGS = [
    "O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC", "B-ORG",
    "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC", "B-PRODUCT", "I-PRODUCT",
    "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-PERCENT", "I-PERCENT", "B-MONEY",
    "I-MONEY", "B-QUANTITY", "I-QUANTITY", "B-ORDINAL", "I-ORDINAL",
    "B-CARDINAL", "I-CARDINAL", "B-EVENT", "I-EVENT", "B-WORK_OF_ART",
    "I-WORK_OF_ART", "B-LAW", "I-LAW", "B-LANGUAGE", "I-LANGUAGE"
]

# NER_TAGS = [
#     'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC',
#     'I-MISC'
# ]


def tokens2wordpiece(tokens,
                     ner_tags,
                     pos_tags,
                     tokenizer,
                     ner_targets=NER_TAGS):
    wp_tokens = []
    wp_ner_tags = []
    wp_pos_tags = []
    for token, ner_tag, pos_tag in zip(tokens, ner_tags, pos_tags):
        word_pieces = tokenizer.tokenize(token)
        if "B-" in ner_targets[ner_tag]:
            broadcast_ner_tags = [ner_tag
                                 ] + [ner_tag + 1] * (len(word_pieces) - 1)
        else:
            broadcast_ner_tags = [ner_tag] * len(word_pieces)

        wp_tokens.extend(word_pieces)
        wp_ner_tags.extend(broadcast_ner_tags)

        broadcast_pos_tags = [pos_tag] * len(word_pieces)
        wp_pos_tags.extend(broadcast_pos_tags)
    return wp_tokens, wp_ner_tags, wp_pos_tags


class VLPTokenClassificationDataset(Dataset):

    def __init__(self, data, tokenizer, max_text_len, training=False):
        data_with_text = []
        start_token_id, end_token_id = tokenizer.convert_tokens_to_ids(
            ['[START]', '[END]'])
        for img_path, tokens, ner_tags, pos_tags in data:
            tokens, ner_tags, pos_tags = tokens2wordpiece(
                tokens, ner_tags, pos_tags, tokenizer)
            tokens = tokenizer.convert_tokens_to_ids(tokens)

            tokens = tokens[:max_text_len - 2]
            ner_tags = ner_tags[:max_text_len - 2]
            pos_tags = pos_tags[:max_text_len - 2]

            tokens = [start_token_id] + tokens + [end_token_id]
            ner_tags = [0] + ner_tags + [0]
            pos_tags = [0] + pos_tags + [0]

            tgt = tokens[:-1]
            tgt_y = tokens[1:]

            ner_tags = ner_tags[1:]
            pos_tags = pos_tags[1:]

            padding_len = max_text_len - len(tgt)

            tokens += [0] * padding_len
            ner_tags += [0] * padding_len
            pos_tags += [0] * padding_len
            tgt += [0] * padding_len
            tgt_y += [0] * padding_len

            tgt = torch.tensor(tgt)
            tgt_y = torch.tensor(tgt_y)

            tgt_mask = make_std_mask(tgt, 0)

            data_with_text.append(
                (img_path, tokens, tgt, tgt_y, tgt_mask, ner_tags, pos_tags))

        self.data = data_with_text
        self.training = training

    def __getitem__(self, index):
        image_path, tokens, tgt, tgt_y, tgt_mask, ner_tags, pos_tags = self.data[
            index]
        # TODO: remove
        image_path = "../notebooks/" + image_path
        image = PIL.Image.open(image_path)

        if self.training and random.random() < 0.5:
            image = PIL.Image.fromarray(np.invert(image))

        image = F.to_tensor(image)

        return image, torch.tensor(tokens), torch.tensor(tgt), torch.tensor(
            tgt_y), torch.tensor(tgt_mask), torch.tensor(
                ner_tags), torch.tensor(pos_tags)

    def __len__(self):
        return len(self.data)