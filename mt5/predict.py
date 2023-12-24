import argparse
import csv
import json
import logging
import math
import nltk
import os
import random
from pathlib import Path

import datasets
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import send_example_telemetry

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Predict the GEC")
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. "
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=None,
        help=(
            "activate beam search and beam num"
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help=(
            "activate top k search and k num"
        ),
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help=(
            "activate top p search and p num"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "activate temperature"
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--output_file", type=str, default=None, help="Where to store the final prediction.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    send_example_telemetry("run_gec_prediction", args)
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
        
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
        
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    identityID = raw_datasets["test"]["id"]
    
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    
    if args.tokenizer_name:
        tokenizer = MT5Tokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = MT5Tokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    
    if args.model_name_or_path:
        model = MT5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
        )
        
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    prefix = args.source_prefix if args.source_prefix is not None else ""
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["test"].column_names
    
    text_column = args.text_column
    if text_column not in column_names:
        raise ValueError(
            f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
        )

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        # targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
    
        return model_inputs
    
    with accelerator.main_process_first():
        # Temporarily set max_target_length for validation.
        max_target_length = args.val_max_target_length
        eval_dataset = raw_datasets["test"].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        
    # Log a few random samples from the training set:
    for index in random.sample(range(len(eval_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )
    
    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]
        # rougeLSum expects newline after each sentence
        preds = ["".join(nltk.sent_tokenize(pred)) for pred in preds]
        return preds
    
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    # Prepare everything with our `accelerator`.
    model, eval_dataloader= accelerator.prepare(
        model, eval_dataloader
    )
    
    logger.info("***** Running Prediction *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(eval_dataset)), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    model.eval()
    gen_kwargs = {
        "max_length": args.val_max_target_length,
    }
    
    if args.beam is not None:
        gen_kwargs["num_beams"] = args.beam
        
    if args.top_k is not None:
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_k"] = args.top_k
        
    if args.top_p is not None:
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_p"] = args.top_p
        
    if args.temperature is not None:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        
    result = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            
            generated_tokens = accelerator.gather_for_metrics(generated_tokens)
            generated_tokens = generated_tokens.cpu().numpy()
            
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            decoded_preds = postprocess_text(decoded_preds)
            result.append(decoded_preds)
            
            
    submission = []
    cnt = 0
    for predicts in result:
        for predict in predicts:
            dict = {}
            dict["output"] = predict
            dict["id"] = identityID[cnt]
            cnt += 1
            submission.append(dict)
            
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, args.output_file), "w") as f:
            json.dump(submission, f, indent=4, ensure_ascii=False)
            
if __name__ == "__main__":
    main()