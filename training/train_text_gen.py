from datetime import datetime
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from nltk import word_tokenize
import numpy as np

from model import VLPForTextRecognition, model_config_factory
from dataset import dataset_factory
from training.eval import evaluate_generation as evaluate
from training.utils import get_tokenizer, get_scheduler, get_model_summary

import warnings

warnings.filterwarnings('ignore')

TRAIN_DEBUG_STEP = 100
TRAIN_EVAL_STEP = 1200


def run_text_gen_training(args):
    current_date_str = datetime.now().strftime('%m%d%Y_%H%M%S')
    # os.makedirs(f"./results/mim_features/{current_date_str}", exist_ok=True)

    tokenizer, vocab_size = get_tokenizer(args)

    config = model_config_factory(args.model_name)
    model = VLPForTextRecognition(**config, vocab_size=vocab_size, dropout=0.0)

    if args.pretrained_model_path:
        pretrained = torch.load(args.pretrained_model_path)["model_state"]
        model.load_state_dict(pretrained, strict=False)

    model.train()

    (train_loader,
     num_steps), test_loader = dataset_factory(args, config, tokenizer)
    # num_steps = len(train_loader)
    # num_steps = len(test_loader)
    optimizer, lr_scheduler = get_scheduler(model, args, config, num_steps)

    get_model_summary(model, args.image_size, args.max_text_len,
                      args.num_channels)

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
        for step, (images, tgt, tgt_y, tgt_mask,
                   tokenized_text) in enumerate(progress_bar):
            images, tgt = images.to(device), tgt.to(device)
            tgt_y, tgt_mask = tgt_y.to(device), tgt_mask.to(device)

            tgt_mask = tgt_mask.squeeze(1)

            logits = model(images, tgt, tgt_mask=tgt_mask)

            loss_fct = nn.CrossEntropyLoss(
                label_smoothing=args.label_smoothing)

            loss = loss_fct(logits.view(-1, vocab_size), tgt_y.view(-1))

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
                        tokenizer.decode(tokenized_text[i].tolist()).replace(
                            ' [PAD] ', '').replace('[PAD]', ''))
                    print("@" * 50)
                    print(
                        tokenizer.decode(tgt[i].tolist()).replace(
                            ' [PAD] ', '').replace('[PAD]', ''))
                    print('!' * 50)
                    print(
                        tokenizer.decode(idx.tolist()).replace(
                            ' [PAD] ', '').replace('[PAD]', ''))
                    print('#' * 100)

            logs = {
                "epoch": epoch + 1,
                "loss": f"{np.mean(training_losses[-500:]):.3f}",
                "lr": lr_scheduler.get_last_lr()[0],
                "step": count
            }

            progress_bar.set_postfix(**logs)

            if not (count % TRAIN_EVAL_STEP):
                eval_loss = evaluate(model,
                                     test_loader,
                                     tokenizer,
                                     device=device,
                                     vocab_size=vocab_size,
                                     debug_steps=50)

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