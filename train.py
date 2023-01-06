import argparse

from training import run_training

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
    parser.add_argument("--dataset_name",
                        type=str,
                        default=None,
                        help="Dataset name")
    parser.add_argument(
        "--dataset_path",  # data/images_ag_news
        type=str,
        default=None,
        help="Path to dataset file")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-cased")
    parser.add_argument("--max_text_len", type=int, default=144)
    parser.add_argument("--out_model_path", type=str, default="./model.pth")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--save_model_steps", type=int, default=3000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_warmup_steps", type=float, default=10000)
    parser.add_argument("--use_clip_grad", type=bool, default=False)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default=None,
                        help="Path to pretrained model")
    parser.add_argument("--logging_dir", type=str, default="logs")

    args = parser.parse_args()

    if not args.model_name or not args.dataset_name:
        raise ValueError(
            "Please specify model name and/or dataset name.\n" \
            "Available model names are [encoder_decoder_sm, encoder_decoder_base]\n" \
            "Available dataset names are [ag_news_text_recognition, wiki_text_recognition, ontonotes]."
        )

    run_training(args)