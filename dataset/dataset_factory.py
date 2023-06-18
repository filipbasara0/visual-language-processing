import pickle
from torch.utils.data import DataLoader
import random

random.seed(42)

from dataset.text_recog_dataset import VLPTextRecognitionMaskedDataset, VLPTextGenerationDataset
from dataset.text_cls_async import get_online_generator as text_cls_generator
from dataset.text_mnli_async import get_online_generator as text_mnli_generator
from dataset.text_recog_masked import get_online_generator as text_recog_generator
from dataset.text_gen_dataset import get_online_generator as text_gen_generator
from dataset.token_cls_dataset import VLPTokenClassificationDataset
from dataset.text_cls_dataset import VLPTextClassificationDataset


def dataset_factory(args, config, tokenizer):
    if args.dataset_name == "text_recognition":
        return _text_recognition(args, config, tokenizer)
    elif args.dataset_name == "ontonotes":
        return _ontonotes_dataset(args, config, tokenizer)
    elif args.dataset_name == "text_cls":
        return _text_cls(args, config, tokenizer)
    elif args.dataset_name == "mnli_text_cls":
        return _mnli_text_cls(args, config, tokenizer)
    elif args.dataset_name == "text_generation":
        return _text_generation_dataset(args, config, tokenizer)
    else:
        raise ValueError(
            f"Dataset name must be one of {['ag_news_text_recognition', 'wiki_text_recognition']}!"
        )


def _load_dataset(args):
    train_data = pickle.load(open(f'{args.dataset_path}/train.pkl', 'rb+'))
    random.shuffle(train_data)
    test_data = pickle.load(open(f'{args.dataset_path}/validation.pkl', 'rb+'))
    return train_data, test_data


def _text_recognition(args, config, tokenizer):
    train_data, test_data = _load_dataset(args)
    print("DATA LEN", len(train_data))
    num_steps = len(train_data) // args.batch_size
    train_loader = text_recog_generator(train_data,
                                        tokenizer,
                                        max_text_len=args.max_text_len,
                                        batch_size=args.batch_size,
                                        training=True)

    test_set = VLPTextRecognitionMaskedDataset(
        test_data,
        tokenizer,
        max_text_len=args.max_text_len,
    )

    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    return (train_loader, num_steps), test_loader


def _text_cls(args, config, tokenizer):
    train_data, test_data = _load_dataset(args)
    num_steps = len(train_data) // args.batch_size
    train_loader = text_cls_generator(train_data,
                                      tokenizer,
                                      max_text_len=args.max_text_len,
                                      batch_size=args.batch_size,
                                      training=True)
    test_set = VLPTextClassificationDataset(test_data)

    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    return (train_loader, num_steps), test_loader


def _mnli_text_cls(args, config, tokenizer):
    train_data, test_data = _load_dataset(args)
    num_steps = len(train_data) // args.batch_size
    train_loader = text_mnli_generator(train_data,
                                       tokenizer,
                                       max_text_len=args.max_text_len,
                                       batch_size=args.batch_size,
                                       training=True)
    test_set = VLPTextClassificationDataset(test_data)

    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    return (train_loader, num_steps), test_loader


def _text_generation_dataset(args, config, tokenizer):
    train_data, test_data = _load_dataset(args)
    num_steps = len(train_data) // args.batch_size
    train_loader = text_gen_generator(train_data,
                                      tokenizer,
                                      max_text_len=args.max_text_len,
                                      batch_size=args.batch_size,
                                      training=True)
    test_set = VLPTextGenerationDataset(
        test_data,
        tokenizer,
        max_text_len=args.max_text_len,
    )

    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    return (train_loader, num_steps), test_loader


def _ontonotes_dataset(args, config, tokenizer):
    train_data, test_data = _load_dataset(args)

    train_set = VLPTokenClassificationDataset(train_data,
                                              tokenizer,
                                              max_text_len=args.max_text_len,
                                              training=True)

    test_set = VLPTokenClassificationDataset(
        test_data,
        tokenizer,
        max_text_len=args.max_text_len,
    )

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    return train_loader, test_loader
