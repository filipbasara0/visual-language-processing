from datetime import datetime
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from nltk import word_tokenize
import numpy as np

from model import VLPForTextRecognition, model_config_factory
from dataset import dataset_factory
from training.eval import evaluate_masked as evaluate
from training.utils import get_tokenizer, get_scheduler, get_model_summary

import warnings

warnings.filterwarnings('ignore')

TRAIN_DEBUG_STEP = 1000
TRAIN_EVAL_STEP = 6000


def run_text_rec_masked_training(args):
    tokenizer, vocab_size = get_tokenizer(args)

    config = model_config_factory(args.model_name)
    model = VLPForTextRecognition(**config, vocab_size=vocab_size, dropout=0.0)

    if args.pretrained_model_path:
        pretrained = torch.load(args.pretrained_model_path)["model_state"]
        model.load_state_dict(pretrained, strict=False)

    model.train()

    get_model_summary(model, args.image_size, args.max_text_len,
                      args.num_channels)

    (train_loader,
     num_steps), test_loader = dataset_factory(args, config, tokenizer)
    optimizer, lr_scheduler = get_scheduler(model, args, config, num_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = torch.cuda.amp.GradScaler()

    count = 0
    for epoch in range(args.num_epochs):
        training_losses = []

        progress_bar = tqdm(train_loader,
                            desc='Training',
                            position=0,
                            total=num_steps,
                            leave=True)
        for step, (images, tgt, tgt_y, tgt_mask,
                   mask_tokens_idxs) in enumerate(progress_bar):
            images, tgt = images.to(device), tgt.to(device)
            tgt_y, tgt_mask = tgt_y.to(device), tgt_mask.to(device)
            mask_tokens_idxs = mask_tokens_idxs.to(device)

            tgt_mask = tgt_mask.squeeze(1)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images, tgt, tgt_mask=tgt_mask)

                loss_fct = nn.CrossEntropyLoss(
                    label_smoothing=args.label_smoothing,
                    # ignore_index=-100,
                    reduction="none")
                # tgt_y[tgt_y != tokenizer.mask_token_id] = -100
                # logits[tgt_y != tokenizer.mask_token_id] = -100
                loss = loss_fct(logits.view(-1, vocab_size), tgt_y.view(-1))
                # print("loss", loss.size())
                # print(loss)
                # print("mask_tokens_idxs", mask_tokens_idxs.size())
                # print(mask_tokens_idxs)
                # print(mask_tokens_idxs == -100)
                # a = mask_tokens_idxs == -100
                # print(a.sum())
                loss = loss.view(mask_tokens_idxs.size(0),
                                 mask_tokens_idxs.size(1))
                # loss[mask_tokens_idxs == -100] *= 10
                # loss_mask = torch.zeros_like(loss)
                # for idx, t in enumerate(mask_tokens_idxs):
                #     loss[idx, t[t!=-1]] *= 5
                # loss_mask[idx, t[t!=-1]] = 1.0
                # loss *= loss_mask
                loss = loss.mean()

            # loss.backward()
            # if args.use_clip_grad:
            # clip_grad_norm_(model.parameters(), 1.0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # optimizer.step()
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
                    text = tokenizer.decode(tgt[i].tolist()).replace(
                        ' [PAD] ',
                        '').replace('[PAD]',
                                    '').replace('[START]',
                                                '').replace('[END]',
                                                            '').strip()
                    # text_tokens = word_tokenize(text)
                    text_tokens = tokenizer.tokenize(text)
                    masked_text = [
                        t if idx not in mask_tokens_idxs[i] else "[MASK]"
                        for idx, t in enumerate(text_tokens)
                    ]
                    print(" ".join(masked_text).replace(' [PAD] ', '').replace(
                        '[PAD]', '').replace(" # # ", "##"))
                    print("@" * 50)
                    print(text)
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
                                     vocab_size=vocab_size)

                # if eval_loss < best_eval_loss:
                torch.save(
                    {
                        'model_state': model.state_dict(),
                        #                         'optimizer_state': optimizer.state_dict(),
                    },
                    args.out_model_path)

                best_eval_loss = eval_loss

                model.train()

    return model, optimizer