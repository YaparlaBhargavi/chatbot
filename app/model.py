import os
import sys
import math
import argparse
import logging
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    set_seed,
)

# Import Wiola specific models to ensure they are registered with AutoClasses
from wiola13m.modeling_wiola import WiolaForCausalLM


# Logging setup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Constants

MODEL_ID = "oscowlai/Wiola13M"
DATASET_ID = "databricks/databricks-dolly-15k"
MAX_SEQ_LEN = 512  # Model's max_position_embeddings



# 1. Prompt formatting

def format_dolly_prompt(example):
    """
    Convert a Dolly dataset example into a structured instruction-following
    prompt that teaches the model the instruction -> response pattern.

    The format handles both examples with and without context.
    """
    instruction = example.get("instruction", "").strip()
    context = example.get("context", "").strip()
    response = example.get("response", "").strip()

    if context:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )
    return prompt



# 2. Dataset preparation

def prepare_dataset(tokenizer, max_length, seed=42):
  
    logger.info(f"Loading dataset: {DATASET_ID}")
    raw_dataset = load_dataset(DATASET_ID, split="train")

    logger.info(f"Dataset loaded: {len(raw_dataset)} examples")
    logger.info(f"Categories: {set(raw_dataset['category'])}")

    # Train / eval split (90/10)
    split = raw_dataset.train_test_split(test_size=0.1, seed=seed)
    train_raw = split["train"]
    eval_raw = split["test"]
    logger.info(f"Train: {len(train_raw)} | Eval: {len(eval_raw)}")

    def tokenize_function(examples):
        """Tokenize and format prompts for causal LM training."""
        prompts = []
        for i in range(len(examples["instruction"])):
            example = {
                "instruction": examples["instruction"][i],
                "context": examples["context"][i],
                "response": examples["response"][i],
            }
            text = format_dolly_prompt(example)
            # Add BOS/EOS tokens
            text = f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"
            prompts.append(text)

        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length",  # Changed from False to "max_length"
            return_attention_mask=True,
        )

        # For causal LM, labels = input_ids (the model learns to predict
        # the next token). Padding tokens will be masked by the collator.
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    logger.info("Tokenizing datasets...")
    train_dataset = train_raw.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=train_raw.column_names,
        desc="Tokenizing train",
    )
    eval_dataset = eval_raw.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=eval_raw.column_names,
        desc="Tokenizing eval",
    )

    # Log sequence length statistics
    train_lengths = [len(x) for x in train_dataset["input_ids"]]
    logger.info(
        f"Train seq lengths - min: {min(train_lengths)}, "
        f"max: {max(train_lengths)}, "
        f"mean: {sum(train_lengths)/len(train_lengths):.0f}, "
        f"truncated: {sum(1 for l in train_lengths if l == max_length)}"
    )

    return train_dataset, eval_dataset



# 3. Training configuration (tuned for best results on a 13M model)

def get_training_args(args):
    """
    Return optimized training arguments for Wiola13M fine-tuning.

    Key choices for best results on a small model:
    - Lower learning rate (2e-4) with cosine schedule for smooth convergence
    - Gradient accumulation to simulate larger effective batch size
    - Weight decay for regularization
    - Warmup steps to stabilize early training
    - fp16/bf16 for faster training when available
    - Early stopping to prevent overfitting
    """
    # Determine compute dtype
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        # --- Training hyperparameters ---
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # --- Optimizer & scheduler ---
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        # --- Mixed precision ---
        fp16=use_fp16,
        bf16=use_bf16,

        # --- Evaluation & saving ---
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # --- Logging ---
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"] + (["wandb"] if args.use_wandb else []),
        run_name=f"wiola13m-dolly-{datetime.now().strftime('%Y%m%d_%H%M%S')}",

        # --- Performance ---
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,  # Not needed for 13M params

        # --- Reproducibility ---
        seed=args.seed,
        data_seed=args.seed,
    )

    return training_args



# 4. Metrics computation

def compute_metrics(eval_pred):
    """Compute perplexity from the evaluation loss."""
    return {}


def preprocess_logits_for_metrics(logits, labels):
    """Avoid storing the full logit tensor - just return argmax predictions."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)



# 5. Main training function

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Wiola13M on Dolly 15k"
    )

    # --- Model & data ---
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--dataset_id", type=str, default=DATASET_ID)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--output_dir", type=str, default="./wiola13m-dolly-finetuned")

    # --- Training hyperparameters ---
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device batch size (default: 16)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4, effective batch=64)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Peak learning rate (default: 2e-4)")

    # --- Logging & evaluation ---
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate every N steps (default: 100)")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps (default: 10)")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")

    # --- Performance ---
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers (default: 2)")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Early stopping patience (default: 5)")

    # --- Reproducibility ---
    parser.add_argument("--seed", type=int, default=42)

    # --- Inference demo ---
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, only run inference demo on output_dir")

    # Clear sys.argv to prevent unrecognized arguments from Colab environment
    sys.argv = ['']
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    logger.info("=" * 70)
    logger.info("   Wiola13M Fine-tuning on Dolly 15k")
    logger.info("=" * 70)

    # ----- Device info -----
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected - training on CPU (will be slow)")

   
    # Load tokenizer & model
    
    logger.info(f"Loading config from {args.model_id}...")
    config = AutoConfig.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    logger.info(f"Loading tokenizer from {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    # Ensure pad token is set (required for batched training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    if not args.skip_training:
        logger.info(f"Loading model from {args.model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters:     {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model dtype:          {next(model.parameters()).dtype}")

        
        # Prepare datasets
       
        train_dataset, eval_dataset = prepare_dataset(
            tokenizer,
            max_length=args.max_seq_len,
            seed=args.seed,
        )

       
        # Data collator
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        
        # Training arguments
       
        training_args = get_training_args(args)

        logger.info(
            f"Effective batch size: "
            f"{args.batch_size * args.gradient_accumulation_steps}"
        )
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Max sequence length: {args.max_seq_len}")
        logger.info(f"FP16: {training_args.fp16} | BF16: {training_args.bf16}")

       
        # Initialize Trainer
        
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
            ),
        ]

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        
        # Train!
        
        logger.info("Starting training...")
        train_result = trainer.train()

        
        # Save best model
        
        logger.info(f"Saving best model to {args.output_dir}...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Log final metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Evaluate
        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        eval_loss = eval_metrics.get("eval_loss", float("inf"))
        perplexity = math.exp(eval_loss) if eval_loss < 100 else float("inf")
        logger.info(f"Final eval loss: {eval_loss:.4f}")
        logger.info(f"Final perplexity: {perplexity:.2f}")

    
    # Inference demo
    
    logger.info("\n" + "=" * 70)
    logger.info("   Inference Demo")
    logger.info("=" * 70)

    # Load the fine-tuned model
    logger.info(f"Loading fine-tuned model from {args.output_dir}...")
    ft_config = AutoConfig.from_pretrained(
        args.output_dir,
        trust_remote_code=True,
    )
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.output_dir,
        config=ft_config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    ft_tokenizer = AutoTokenizer.from_pretrained(
        args.output_dir,
        trust_remote_code=True,
    )
    ft_model.eval()

    if torch.cuda.is_available():
        ft_model = ft_model.to("cuda")

    # Test prompts covering different Dolly categories
    test_prompts = [
        "### Instruction:\nExplain what machine learning is in simple terms.\n\n### Response:\n",
        "### Instruction:\nWhat are three benefits of regular exercise?\n\n### Response:\n",
        "### Instruction:\nWrite a short poem about the ocean.\n\n### Response:\n",
        "### Instruction:\nSummarize the concept of supply and demand.\n\n### Response:\n",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        inputs = ft_tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = ft_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=ft_tokenizer.pad_token_id,
                eos_token_id=ft_tokenizer.eos_token_id,
            )

        generated = ft_tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"\n--- Test {i} ---\n{generated}\n")

    logger.info("=" * 70)
    logger.info("   Training complete!")
    logger.info(f"   Model saved to: {os.path.abspath(args.output_dir)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()