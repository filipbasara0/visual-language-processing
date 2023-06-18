from datetime import datetime
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torchinfo import summary

import numpy as np

from model import VLPForTextMLM, model_config_factory
from dataset import dataset_factory
from training.utils import get_tokenizer, get_scheduler

import warnings

warnings.filterwarnings('ignore')

TRAIN_DEBUG_STEP = 1000
TRAIN_EVAL_STEP = 4000


def get_model_summary(model, image_size, num_channels=3):
    return str(
        summary(model, (1, num_channels, image_size, image_size), verbose=1))


def run_text_rec_enc_training(args):
    current_date_str = datetime.now().strftime('%m%d%Y_%H%M%S')
    # os.makedirs(f"./results/mim_features/{current_date_str}", exist_ok=True)

    tokenizer, vocab_size = get_tokenizer(args)

    config = model_config_factory(args.model_name)
    model = VLPForTextMLM(**config, num_classes=vocab_size, dropout=0.0)

    if args.pretrained_model_path:
        pretrained = torch.load(args.pretrained_model_path)["model_state"]
        model.load_state_dict(pretrained, strict=False)

    model.train()

    (train_loader,
     num_steps), test_loader = dataset_factory(args, config, tokenizer)
    # num_steps = len(train_loader)
    # num_steps = len(test_loader)
    optimizer, lr_scheduler = get_scheduler(model, args, config, num_steps)

    get_model_summary(model, args.image_size, args.num_channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_eval_loss = np.inf
    count = 0
    for epoch in range(args.num_epochs):
        training_losses = []

        progress_bar = tqdm(train_loader,
                            desc='Training',
                            position=0,
                            total=num_steps,
                            leave=True)
        for step, (images, target) in enumerate(progress_bar):
            images, target = images.to(device), target.to(device)

            logits = model(images)

            loss_fct = nn.CrossEntropyLoss(
                label_smoothing=args.label_smoothing)

            logits = logits[:, :args.max_text_len, :]
            loss = loss_fct(logits.reshape(-1, vocab_size), target.view(-1))

            loss.backward()

            if args.use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            training_losses.append(loss.item())
            count += 1

            if step % TRAIN_DEBUG_STEP == 0:
                idxs = torch.argmax(logits, dim=-1)
                for i, idx in enumerate(idxs):
                    if i == 1:
                        break
                    print('#' * 100)
                    print(
                        tokenizer.decode(target[i].tolist()).replace(
                            ' [PAD] ', '').replace('[PAD]', ''))
                    print('!' * 50)
                    print(
                        tokenizer.decode(idx.tolist()).replace(
                            ' [PAD] ', '').replace('[PAD]', ''))
                    print('#' * 100)

            logs = {
                "epoch": epoch + 1,
                "loss": f"{np.mean(training_losses[-2000:]):.3f}",
                "lr": lr_scheduler.get_last_lr()[0],
                "step": count
            }

            progress_bar.set_postfix(**logs)

            if not (count % TRAIN_EVAL_STEP):
                eval_loss = evaluate(model,
                                     test_loader,
                                     tokenizer,
                                     device=device,
                                     max_text_len=args.max_text_len,
                                     vocab_size=vocab_size)

                if eval_loss < best_eval_loss:
                    torch.save(
                        {
                            'model_state': model.state_dict(),
                            #                         'optimizer_state': optimizer.state_dict(),
                        },
                        args.out_model_path)

                    best_eval_loss = eval_loss

                model.train()

    return model, optimizer


def evaluate(model, test_loader, tokenizer, vocab_size, max_text_len, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for step, (images, tgt) in enumerate(tqdm(test_loader)):
            images, tgt = images.to(device), tgt.to(device)

            logits = model(images)

            loss_fct = nn.CrossEntropyLoss()
            logits = logits[:, :max_text_len, :]
            loss = loss_fct(logits.reshape(-1, vocab_size), tgt.view(-1))

            if step % 250 == 0:
                idxs = torch.argmax(logits, dim=-1)
                for i, idx in enumerate(idxs):
                    if i == 1:
                        break
                    print('#' * 100)
                    print(
                        tokenizer.decode(tgt[i].tolist()).replace(
                            ' [PAD] ', '').replace('[PAD]', ''))
                    print('!' * 50)
                    print(
                        tokenizer.decode(idx.tolist()).replace(
                            ' [PAD] ', '').replace('[PAD]', ''))
                    print('#' * 100)

            total_loss += loss.item()

        total_loss /= len(test_loader)
        print(f"Valid Loss: {total_loss}")
    return total_loss