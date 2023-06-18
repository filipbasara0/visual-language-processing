from .train_text_rec import run_text_rec_training
from .train_token_cls import run_token_cls_training
from .train_text_cls import run_text_cls_training
from .train_text_rec_enc import run_text_rec_enc_training
from .train_text_rec_masked import run_text_rec_masked_training
from .train_text_gen import run_text_gen_training

VALID_TASKS = [
    "text_recognition", "token_classification", "text_classification",
    "text_recognition_encoder", "text_recognition_masked", "text_generation"
]


def run_training(args):
    if args.task_name == "text_recognition":
        run_text_rec_training(args)
    elif args.task_name == "token_classification":
        run_token_cls_training(args)
    elif args.task_name == "text_classification":
        run_text_cls_training(args)
    elif args.task_name == "text_recognition_encoder":
        run_text_rec_enc_training(args)
    elif args.task_name == "text_recognition_masked":
        run_text_rec_masked_training(args)
    elif args.task_name == "text_generation":
        run_text_gen_training(args)
    else:
        raise ValueError(
            f"Please select one of following tasks: {VALID_TASKS}")
