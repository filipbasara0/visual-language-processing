import PIL
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import Resize

from inference.decode import greedy_decode


def predict(
    model,
    image_path,
    tokenizer,
    max_text_len,
    image_size=512,
    device=None,
):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = PIL.Image.open(image_path).convert("RGB")
    image = Resize((image_size, image_size))(image)
    image = F.to_tensor(image).to(device).unsqueeze(0)
    out = greedy_decode(model, image, max_text_len,
                        tokenizer.convert_tokens_to_ids(["[START]"])[0])
    return out
