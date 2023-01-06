from .train_text_rec import run_text_rec_training
from .train_token_cls import run_token_cls_training

VALID_TASKS = ["text_recognition", "token_classification"]


def run_training(args):
    if args.task_name == "text_recognition":
        run_text_rec_training(args)
    elif args.task_name == "token_classification":
        run_token_cls_training(args)
    else:
        raise ValueError(f"Please select one of following tasks: {VALID_TASKS}")
