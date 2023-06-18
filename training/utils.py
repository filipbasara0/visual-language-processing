import torch
from torch.optim import AdamW
from torchinfo import summary
from transformers import AutoTokenizer

from model.utils import subsequent_mask


def get_scheduler(model, args, config, steps_per_epoch):
    optimizer = AdamW(model.parameters(),
                      lr=args.learning_rate,
                      betas=(0.9, 0.99),
                      eps=1e-8,
                      weight_decay=5e-2)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=args.num_epochs)

    return optimizer, lr_scheduler


def get_model_summary(model, image_size, max_text_len, num_channels=3):
    img = torch.rand(1, num_channels, image_size, image_size)
    tgt = torch.randint(high=20000, size=(
        1,
        max_text_len,
    ))
    tgt_mask = subsequent_mask(max_text_len).long()
    return str(summary(model, input_data=[img, tgt, tgt_mask], verbose=1))


def get_tokenizer(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    special_tokens_dict = {'additional_special_tokens': ['[START]', '[END]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    vocab_size = tokenizer.vocab_size + num_added_toks

    return tokenizer, vocab_size
