import torch
from model.utils import subsequent_mask


def greedy_decode(model, image, max_len, start_symbol):
    device = image.device
    features = model.vlp.get_image_features(image)
    memory = model.vlp.transformer.encode(features)
    ys = torch.ones(1, 1).fill_(start_symbol).long().to(device)
    for _ in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1)).long().to(device)
        out = model.vlp.transformer.decode(memory, ys, tgt_mask)
        prob = model.cls(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.ones(1, 1).long().fill_(next_word).to(device)],
            dim=1).to(device)
    return ys