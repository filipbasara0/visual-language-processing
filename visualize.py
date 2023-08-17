import argparse
import glob
import os
from tqdm.auto import tqdm
import torch

from model import VLPForTextClassification, model_config_factory
from inference.integrated_gradients import integrate_gradients


def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = model_config_factory(args.model_name)
    model = VLPForTextClassification(**config,
                                     num_classes=args.num_classes,
                                     dropout=0.0)
    model_state = torch.load(args.pretrained_model_path)["model_state"]
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)
    image_files = glob.glob(args.images_path)
    image_files = [f for f in image_files if os.path.isfile(f)]

    integrate_gradients(model, path = args.images_path)


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
    parser.add_argument("--tokenizer_name",
                        type=str,
                        default="bert-base-cased")
    parser.add_argument("--max_text_len", type=int, default=144)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=3)
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

    visualize(args)