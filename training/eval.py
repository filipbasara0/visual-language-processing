import torch
import torch.nn as nn
from tqdm.auto import tqdm


def evaluate(model, test_loader, tokenizer, vocab_size, device, debug_steps=300):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for step, (images, tgt, tgt_y,
                   tgt_mask) in enumerate(tqdm(test_loader)):

            images, tgt = images.to(device), tgt.to(device)
            tgt_y, tgt_mask = tgt_y.to(device), tgt_mask.to(device)
            tgt_mask = tgt_mask.squeeze(1)

            logits = model(images, tgt, tgt_mask=tgt_mask)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, vocab_size), tgt_y.view(-1))

            if step % debug_steps == 0:
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
                        tokenizer.decode(idx.tolist()).replace(' [PAD] ',
                                                               '').replace(
                                                                   '[PAD]', ''))
                    print('#' * 100)

            total_loss += loss.item()

        total_loss /= len(test_loader)
        print(f"Valid Loss: {total_loss}")
    return total_loss


def evaluate_masked(model, test_loader, tokenizer, vocab_size, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for step, (images, tgt, tgt_y, tgt_mask,
                   masked_tokens) in enumerate(tqdm(test_loader)):
            # if step == 10: break
            images, tgt = images.to(device), tgt.to(device)
            tgt_y, tgt_mask = tgt_y.to(device), tgt_mask.to(device)
            tgt_mask = tgt_mask.squeeze(1)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images, tgt, tgt_mask=tgt_mask)

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, vocab_size), tgt_y.view(-1))

            if step % 300 == 0:
                idxs = torch.argmax(logits, dim=-1)
                for i, idx in enumerate(idxs):
                    if i == 1:
                        break
                    print('#' * 100)
                    print(
                        tokenizer.decode(masked_tokens[i]).replace(
                            ' [PAD] ', '').replace('[PAD]', ''))
                    print("@" * 50)
                    print(
                        tokenizer.decode(tgt[i].tolist()).replace(
                            ' [PAD] ', '').replace('[PAD]', ''))
                    print('!' * 50)
                    print(
                        tokenizer.decode(idx.tolist()).replace(' [PAD] ',
                                                               '').replace(
                                                                   '[PAD]', ''))
                    print('#' * 100)

            total_loss += loss.item()

        total_loss /= len(test_loader)
        print(f"Valid Loss: {total_loss}")
    return total_loss


def evaluate_generation(model, test_loader, tokenizer, vocab_size, device, debug_steps=100):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for step, (images, tgt, tgt_y, tgt_mask,
                   tokenized_text) in enumerate(tqdm(test_loader)):

            images, tgt = images.to(device), tgt.to(device)
            tgt_y, tgt_mask = tgt_y.to(device), tgt_mask.to(device)
            tgt_mask = tgt_mask.squeeze(1)

            logits = model(images, tgt, tgt_mask=tgt_mask)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, vocab_size), tgt_y.view(-1))

            if step % debug_steps == 0:
                idxs = torch.argmax(logits, dim=-1)
                for i, idx in enumerate(idxs):
                    if i == 1:
                        break
                    print('#' * 100)
                    print(
                        tokenizer.decode(tokenized_text[i]).replace(
                            ' [PAD] ', '').replace('[PAD]', ''))
                    print("@" * 50)
                    print(
                        tokenizer.decode(tgt[i].tolist()).replace(
                            ' [PAD] ', '').replace('[PAD]', ''))
                    print('!' * 50)
                    print(
                        tokenizer.decode(idx.tolist()).replace(' [PAD] ',
                                                               '').replace(
                                                                   '[PAD]', ''))
                    print('#' * 100)

            total_loss += loss.item()

        total_loss /= len(test_loader)
        print(f"Valid Loss: {total_loss}")
    return total_loss