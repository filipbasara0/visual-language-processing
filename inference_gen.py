import argparse
import glob
import os
from tqdm.auto import tqdm
import torch

from model import VLPForTextRecognition, model_config_factory
from training.utils import get_tokenizer
from inference.predict import predict


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, vocab_size = get_tokenizer(args)

    config = model_config_factory(args.model_name)
    model = VLPForTextRecognition(**config, vocab_size=vocab_size, dropout=0.0)
    model_state = torch.load(args.pretrained_model_path)["model_state"]
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)
    image_files = glob.glob(args.images_path)
    image_files = [f for f in image_files if os.path.isfile(f)]

    for idx, image_path in enumerate(tqdm(image_files,
                                          total=len(image_files))):
        out = predict(
            model,
            image_path,
            tokenizer,
            args.max_text_len,
            image_size=args.image_size,
            device=device,
        )
        print(
            idx, image_path,
            tokenizer.decode(out[0].tolist()).replace(' [PAD] ',
                                                      '').replace('[PAD]', ''))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLP training script")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Task name - one of [text_recognition, token_classification] ")
    parser.add_argument("--model_name",
                        type=str,
                        default=None,
                        help="Model name")
    parser.add_argument("--images_path",
                        type=str,
                        default=None,
                        help="Path to dataset file")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--tokenizer_name",
                        type=str,
                        default="bert-base-cased")
    parser.add_argument("--max_text_len", type=int, default=144)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default=None,
                        help="Path to pretrained model")
    parser.add_argument("--logging_dir", type=str, default="logs")

    args = parser.parse_args()

    if not args.model_name or not args.pretrained_model_path:
        raise ValueError(
            "Please specify model name and/or pretrained model path.\n" \
            "Available model names are [encoder_decoder_sm, encoder_decoder_base]\n" \
        )

    run_inference(args)