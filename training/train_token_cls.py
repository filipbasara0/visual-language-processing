from datetime import datetime
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from sklearn import metrics
import numpy as np

from model import VLPForTokenClassification, model_config_factory
from dataset import dataset_factory
from training.utils import get_tokenizer, get_scheduler, get_model_summary

import warnings

warnings.filterwarnings('ignore')

POS_TAGS = [
    "XX", "``", "$", "''", "*", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX",
    "CC", "CD", "DT", "EX", "FW", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD",
    "NFP", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR",
    "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "VERB", "WDT", "WP", "WP$", "WRB"
]
NER_TAGS = [
    "O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC", "B-ORG",
    "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC", "B-PRODUCT", "I-PRODUCT",
    "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-PERCENT", "I-PERCENT", "B-MONEY",
    "I-MONEY", "B-QUANTITY", "I-QUANTITY", "B-ORDINAL", "I-ORDINAL",
    "B-CARDINAL", "I-CARDINAL", "B-EVENT", "I-EVENT", "B-WORK_OF_ART",
    "I-WORK_OF_ART", "B-LAW", "I-LAW", "B-LANGUAGE", "I-LANGUAGE"
]

# POS_TAGS = [
#     'XX', '"', "''", '#', '$', '(', ')', ',', '.', ':', '``', 'CC', 'CD', 'DT',
#     'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS',
#     'NNS', 'NN|SYM', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
#     'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
#     'WP$', 'WRB'
# ]

# NER_TAGS = [
#     'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC',
#     'I-MISC'
# ]

TRAIN_DEBUG_STEP = 1000
TRAIN_EVAL_STEP = 3000

# TRAIN_DEBUG_STEP = 500
# TRAIN_EVAL_STEP = 1000


def run_token_cls_training(args):
    current_date_str = datetime.now().strftime('%m%d%Y_%H%M%S')
    # os.makedirs(f"./results/mim_features/{current_date_str}", exist_ok=True)

    tokenizer, vocab_size = get_tokenizer(args)

    config = model_config_factory(args.model_name)
    model = VLPForTokenClassification(
        model_dim=config["model_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        feature_map_size=config["feature_map_size"],
        vocab_size=vocab_size,
        num_ner_tags=len(NER_TAGS),
        num_pos_tags=len(POS_TAGS),
        dropout=0.0)

    if args.pretrained_model_path:
        pretrained = torch.load(args.pretrained_model_path)["model_state"]
        model.load_state_dict(pretrained, strict=False)

    model.train()

    train_loader, test_loader = dataset_factory(args, config, tokenizer)
    optimizer, lr_scheduler = get_scheduler(model, args, config,
                                            len(train_loader))

    get_model_summary(model, args.image_size, args.max_text_len,
                      args.num_channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_eval_loss = np.inf
    count = 0
    for epoch in range(args.num_epochs):
        training_losses = []
        tr_ner_losses = []
        tr_pos_losses = []
        tr_txt_losses = []

        f1_micro_scores_ner, f1_macro_scores_ner = [], []
        f1_micro_scores_pos, f1_macro_scores_pos = [], []
        progress_bar = tqdm(train_loader,
                            desc='Training',
                            position=0,
                            leave=True)
        for step, (images, tokens, tgt, tgt_y, tgt_mask, ner_tags,
                   pos_tags) in enumerate(progress_bar):
            images, tgt = images.to(device), tgt.to(device)
            tgt_y, tgt_mask = tgt_y.to(device), tgt_mask.to(device)
            ner_tags, pos_tags = ner_tags.to(device), pos_tags.to(device)

            tgt_mask = tgt_mask.squeeze(1)

            logits_txt, logits_ner, logits_pos = model(images,
                                                       tgt,
                                                       tgt_mask=tgt_mask)

            loss_fct = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            loss_txt = loss_fct(logits_txt.view(-1, vocab_size), tgt_y.view(-1))
            loss_ner = loss_fct(logits_ner.view(-1, len(NER_TAGS)),
                                ner_tags.view(-1))
            loss_pos = loss_fct(logits_pos.view(-1, len(POS_TAGS)),
                                pos_tags.view(-1))
            loss = loss_txt + loss_ner + loss_pos

            loss.backward()
            if args.use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            training_losses.append(loss.item())
            tr_txt_losses.append(loss_txt.item())
            tr_ner_losses.append(loss_ner.item())
            tr_pos_losses.append(loss_pos.item())
            count += 1

            preds_ner = logits_ner.argmax(dim=-1)
            preds_pos = logits_pos.argmax(dim=-1)

            target_ner, preds_ner = ner_tags.cpu().detach().numpy(
            ), preds_ner.cpu().detach().numpy()
            target_pos, preds_pos = pos_tags.cpu().detach().numpy(
            ), preds_pos.cpu().detach().numpy()

            f1_score_micro_ner, f1_score_macro_ner = get_metrics(
                target_ner.flatten(), preds_ner.flatten(), display=False)
            f1_micro_scores_ner.append(f1_score_micro_ner)
            f1_macro_scores_ner.append(f1_score_macro_ner)

            f1_score_micro_pos, f1_score_macro_pos = get_metrics(
                target_pos.flatten(), preds_pos.flatten(), display=False)
            f1_micro_scores_pos.append(f1_score_micro_pos)
            f1_macro_scores_pos.append(f1_score_macro_pos)

            if step % TRAIN_DEBUG_STEP == 0:
                print("train target_ner", [NER_TAGS[t] for t in target_ner[0]])
                print("train preds_ner", [NER_TAGS[t] for t in preds_ner[0]])
                print("!" * 100)
                print("train target_pos", [POS_TAGS[t] for t in target_pos[0]])
                print("train preds_pos", [POS_TAGS[t] for t in preds_pos[0]])
                print("#" * 50)
                idxs = torch.argmax(logits_txt, dim=-1)
                for i, idx in enumerate(idxs):
                    if i == 1:
                        break
                    print('#' * 100)
                    print(
                        tokenizer.decode(tokens[i].tolist()).replace(
                            ' [PAD] ', '').replace('[PAD]', ''))
                    print('!' * 50)
                    print(
                        tokenizer.decode(idx.tolist()).replace(' [PAD] ',
                                                               '').replace(
                                                                   '[PAD]', ''))
                    print('#' * 100)

            logs = {
                "epoch": epoch + 1,
                # "loss": f"{np.mean(training_losses[-2000:]):.3f}",
                "loss_ner": f"{np.mean(tr_ner_losses[-2000:]):.3f}",
                "loss_pos": f"{np.mean(tr_pos_losses[-2000:]):.3f}",
                "loss_txt": f"{np.mean(tr_txt_losses[-2000:]):.3f}",
                # "f1_micro_ner": f"{np.mean(f1_micro_scores_ner[-2000:]):.3f}",
                "f1_ner:": f"{np.mean(f1_macro_scores_ner[-2000:]):.3f}",
                # "f1_micro_pos": f"{np.mean(f1_micro_scores_pos[-2000:]):.3f}",
                "f1_pos:": f"{np.mean(f1_macro_scores_pos[-2000:]):.3f}",
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
                                     epoch=epoch,
                                     tr_step=step,
                                     current_date_str=current_date_str)

                if eval_loss < best_eval_loss:
                    torch.save(
                        {
                            'model_state': model.state_dict(),
                            #                         'optimizer_state': optimizer.state_dict(),
                        },
                        args.out_model_path)

                    best_eval_loss = eval_loss

                model.train()

    torch.save({
        'model_state': model.state_dict(),
    }, f"last_{args.out_model_path}")

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


def evaluate(model,
             test_loader,
             tokenizer,
             vocab_size,
             device,
             epoch=0,
             tr_step=0,
             current_date_str="dummy_date"):
    print("usao")
    model.eval()
    total_loss_ner = 0
    total_loss_pos = 0
    total_loss_txt = 0
    total_loss = 0
    all_targets_ner = []
    all_preds_ner = []
    all_targets_pos = []
    all_preds_pos = []
    with torch.no_grad():
        for step, (images, tokens, tgt, tgt_y, tgt_mask, ner_tags,
                   pos_tags) in enumerate(tqdm(test_loader)):

            images, tgt = images.to(device), tgt.to(device)
            tgt_y, tgt_mask = tgt_y.to(device), tgt_mask.to(device)
            ner_tags, pos_tags = ner_tags.to(device), pos_tags.to(device)

            tgt_mask = tgt_mask.squeeze(1)

            logits_txt, logits_ner, logits_pos = model(images,
                                                       tgt,
                                                       tgt_mask=tgt_mask)

            loss_fct = nn.CrossEntropyLoss()
            loss_txt = loss_fct(logits_txt.view(-1, vocab_size), tgt_y.view(-1))
            loss_ner = loss_fct(logits_ner.view(-1, len(NER_TAGS)),
                                ner_tags.view(-1))
            loss_pos = loss_fct(logits_pos.view(-1, len(POS_TAGS)),
                                pos_tags.view(-1))
            loss = loss_txt + loss_ner + loss_pos

            preds_ner = logits_ner.argmax(dim=-1)
            preds_pos = logits_pos.argmax(dim=-1)

            target_ner, preds_ner = ner_tags.cpu().detach().numpy(
            ), preds_ner.cpu().detach().numpy()
            target_pos, preds_pos = pos_tags.cpu().detach().numpy(
            ), preds_pos.cpu().detach().numpy()
            all_targets_ner.extend(target_ner.flatten())
            all_preds_ner.extend(preds_ner.flatten())
            all_targets_pos.extend(target_pos.flatten())
            all_preds_pos.extend(preds_pos.flatten())

            if step % 300 == 0:
                print("eval target_ner", [NER_TAGS[t] for t in target_ner[0]])
                print("eval preds_ner", [NER_TAGS[t] for t in preds_ner[0]])
                print("!" * 100)
                print("eval target_pos", [POS_TAGS[t] for t in target_pos[0]])
                print("eval preds_pos", [POS_TAGS[t] for t in preds_pos[0]])
                print("#" * 50)
                idxs = torch.argmax(logits_txt, dim=-1)
                for i, idx in enumerate(idxs):
                    if i == 1:
                        break
                    print('#' * 100)
                    print(
                        tokenizer.decode(tokens[i].tolist()).replace(
                            ' [PAD] ', '').replace('[PAD]', ''))
                    print('!' * 50)
                    print(
                        tokenizer.decode(idx.tolist()).replace(' [PAD] ',
                                                               '').replace(
                                                                   '[PAD]', ''))
                    print('#' * 100)

            total_loss += loss.item()
            total_loss_txt += loss_txt.item()
            total_loss_ner += loss_ner.item()
            total_loss_pos += loss_pos.item()

        total_loss /= len(test_loader)
        total_loss_txt /= len(test_loader)
        total_loss_ner /= len(test_loader)
        total_loss_pos /= len(test_loader)

        print(f"Valid Loss: {total_loss}")
        print(f"Valid Loss Text: {total_loss_txt}")
        print(f"Valid Loss NER: {total_loss_ner}")
        print(f"Valid Loss POS: {total_loss_pos}")
        get_metrics(all_targets_ner, all_preds_ner, display=True)
        print(
            metrics.classification_report(all_targets_ner,
                                          all_preds_ner,
                                          target_names=NER_TAGS,
                                          labels=list(range(len(NER_TAGS)))))
        get_metrics(all_targets_pos, all_preds_pos, display=True)
        print(
            metrics.classification_report(all_targets_pos,
                                          all_preds_pos,
                                          target_names=POS_TAGS,
                                          labels=list(range(len(POS_TAGS)))))
    return total_loss
