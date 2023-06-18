from datetime import datetime
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torchinfo import summary

import numpy as np
from sklearn import metrics

from model import VLPForTextClassification, model_config_factory
from dataset import dataset_factory
from training.utils import get_tokenizer, get_scheduler

import warnings

warnings.filterwarnings('ignore')

TRAIN_DEBUG_STEP = 500
TRAIN_EVAL_STEP = 2000

# TRAIN_DEBUG_STEP = 1000
# TRAIN_EVAL_STEP = 4000


def get_model_summary(model, image_size, num_channels=3):
    return str(
        summary(model, (1, num_channels, image_size, image_size), verbose=1))


def run_text_cls_training(args):
    tokenizer, _ = get_tokenizer(args)

    num_classes = args.num_classes
    config = model_config_factory(args.model_name)
    model = VLPForTextClassification(**config,
                                     num_classes=num_classes,
                                     dropout=0.0)

    if args.pretrained_model_path:
        pretrained = torch.load(args.pretrained_model_path)["model_state"]
        model.load_state_dict(pretrained, strict=False)

    model.train()

    (train_loader,
     num_steps), test_loader = dataset_factory(args, config, tokenizer)
    optimizer, lr_scheduler = get_scheduler(model, args, config, num_steps)

    get_model_summary(model, args.image_size, args.num_channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    best_eval_loss = np.inf
    count = 0
    for epoch in range(args.num_epochs):
        training_losses = []

        progress_bar = tqdm(train_loader,
                            desc='Training',
                            position=0,
                            total=num_steps,
                            leave=True)
        f1_micro_scores, f1_macro_scores = [], []
        for step, (images, targets) in enumerate(progress_bar):
            images, targets = images.to(device), targets.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)

                loss_fct = nn.CrossEntropyLoss(
                    label_smoothing=args.label_smoothing)

                loss = loss_fct(logits, targets)

            # loss.backward()
            # if args.use_clip_grad:
            #     clip_grad_norm_(model.parameters(), 1.0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            preds = logits.argmax(dim=-1)
            fin_targets, fin_outputs = targets.cpu().detach().numpy(
            ), preds.cpu().detach().numpy()
            f1_score_micro, f1_score_macro = get_metrics(fin_targets,
                                                         fin_outputs,
                                                         display=False)
            f1_micro_scores.append(f1_score_micro)
            f1_macro_scores.append(f1_score_macro)

            training_losses.append(loss.item())
            count += 1

            if step % TRAIN_DEBUG_STEP == 0:
                print("train fin_targets", fin_targets)
                print("train fin_outputs", fin_outputs)
                print("#" * 50)

            logs = {
                "epoch": epoch + 1,
                "loss": f"{np.mean(training_losses[-2000:]):.3f}",
                "f1_micro": f"{np.mean(f1_micro_scores[-2000:]):.3f}",
                "f1_macro:": f"{np.mean(f1_macro_scores[-2000:]):.3f}",
                "lr": lr_scheduler.get_last_lr()[0],
                "step": count
            }

            progress_bar.set_postfix(**logs)

            if not (count % TRAIN_EVAL_STEP):
                eval_loss = evaluate(model, test_loader, device=device)

                if eval_loss < best_eval_loss:
                    torch.save(
                        {
                            'model_state': model.state_dict(),
                            #                         'optimizer_state': optimizer.state_dict(),
                        },
                        args.out_model_path)

                    best_eval_loss = eval_loss

                model.train()

        # lr_scheduler.step()
    return model, optimizer


def get_metrics(targets, outputs, display=True):
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    if display:
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
    return f1_score_micro, f1_score_macro


def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for step, (images, targets) in enumerate(tqdm(test_loader)):
            # if step == 100:
            #     break
            images, targets = images.to(device), targets.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, targets)

            preds = logits.argmax(dim=-1)
            fin_targets.extend(targets.cpu().detach().numpy())
            fin_outputs.extend(preds.cpu().detach().numpy())

            if step % 200 == 0:
                print("eval fin_targets", targets.cpu().detach().numpy())
                print("eval fin_outputs", preds.cpu().detach().numpy())
                print("#" * 50)

            total_loss += loss.item()

        total_loss /= len(test_loader)
        print(f"Valid Loss: {total_loss}")
        get_metrics(fin_targets, fin_outputs)
        print(metrics.classification_report(fin_targets, fin_outputs))
    return total_loss