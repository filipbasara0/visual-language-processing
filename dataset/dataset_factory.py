import pickle
from torch.utils.data import DataLoader
import random

random.seed(42)

from dataset.text_recog_dataset import VLPTextRecognitionDataset
from dataset.text_recog_online_dataset import VLPDatasetOnline
# from dataset.text_recog_online_async_dataset import get_online_generator
from dataset.txt_rec_online_async_for import get_online_generator
from dataset.token_cls_dataset import VLPTokenClassificationDataset


def dataset_factory(args, config, tokenizer):
    if args.dataset_name == "ag_news_text_recognition":
        return _ag_news_text_recognition(args, config, tokenizer)
    elif args.dataset_name == "wiki_text_recognition":
        # return _wiki_text_recognition(args, config, tokenizer)
        return _wiki_text_recognition_async(args, config, tokenizer)
    elif args.dataset_name == "ontonotes":
        return _ontonotes_dataset(args, config, tokenizer)
    else:
        raise ValueError(
            f"Dataset name must be one of {['ag_news_text_recognition', 'wiki_text_recognition']}!"
        )


def _load_dataset(args):
    train_data = pickle.load(open(
        f'{args.dataset_path}/train.pkl', 'rb+')) + pickle.load(
            open('../notebooks/data/images_books/train_sm.pkl', 'rb+'))
    random.shuffle(train_data)
    test_data = pickle.load(
        open(f'{args.dataset_path}/validation.pkl', 'rb+')
    )  # + pickle.load(open('../notebooks/data/images_books/validation.pkl', 'rb+'))
    return train_data, test_data


def _ag_news_text_recognition(args, config, tokenizer):
    train_data, test_data = _load_dataset(args)

    train_set = VLPTextRecognitionDataset(train_data,
                                          tokenizer,
                                          max_text_len=config["max_text_len"],
                                          training=True)

    test_set = VLPTextRecognitionDataset(
        test_data,
        tokenizer,
        max_text_len=config["max_text_len"],
    )

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    return train_loader, test_loader


def _wiki_text_recognition(args, config, tokenizer):
    train_data, test_data = _load_dataset(args)

    train_set = VLPDatasetOnline(train_data,
                                 tokenizer,
                                 max_text_len=config["max_text_len"],
                                 training=True)

    test_set = VLPTextRecognitionDataset(
        test_data,
        tokenizer,
        max_text_len=config["max_text_len"],
    )

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    return train_loader, test_loader


def _wiki_text_recognition_async(args, config, tokenizer):
    train_data, test_data = _load_dataset(args)
    print("DATA LEN", len(train_data))
    train_data = train_data[::-1]
    num_steps = len(train_data) // args.batch_size
    train_loader = get_online_generator(train_data,
                                        tokenizer,
                                        max_text_len=config["max_text_len"],
                                        batch_size=args.batch_size,
                                        training=True)

    #########################################################################
    # train_set = VLPDatasetOnline(train_data,
    #                              tokenizer,
    #                              max_text_len=config["max_text_len"],
    #                              training=True)
    # train_loader = DataLoader(train_set,
    #                           batch_size=args.batch_size,
    #                           num_workers=8,
    #                           shuffle=True)
    # num_steps = len(train_loader)
    #########################################################################

    test_set = VLPTextRecognitionDataset(
        test_data,
        tokenizer,
        max_text_len=config["max_text_len"],
    )

    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    return (train_loader, num_steps), test_loader


def _ontonotes_dataset(args, config, tokenizer):
    train_data, test_data = _load_dataset(args)

    train_set = VLPTokenClassificationDataset(
        train_data,
        tokenizer,
        max_text_len=config["max_text_len"],
        training=True)

    test_set = VLPTokenClassificationDataset(
        test_data,
        tokenizer,
        max_text_len=config["max_text_len"],
    )

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    return train_loader, test_loader
