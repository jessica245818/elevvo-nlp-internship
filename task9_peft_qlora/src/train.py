"""Fine-tune FLAN-T5 with QLoRA on CUDA or LoRA on other devices."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments


MODEL_ID = "google/flan-t5-small"


class DialogueDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: object):
        inputs = tokenizer([f"summarize dialogue: {text}" for text in frame.dialogue], max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        labels = tokenizer(frame.summary.tolist(), max_length=96, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        self.items = {**inputs, "labels": labels}

    def __len__(self) -> int:
        return len(self.items["input_ids"])

    def __getitem__(self, index: int) -> dict:
        return {key: value[index] for key, value in self.items.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("task9_peft_qlora/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("task9_peft_qlora/artifacts"))
    parser.add_argument("--train-examples", type=int, default=2000)
    parser.add_argument("--validation-examples", type=int, default=200)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--force-lora", action="store_true", help="Disable 4-bit loading even on CUDA")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_qlora = torch.cuda.is_available() and not args.force_lora
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model_kwargs = {}
    if use_qlora:
        model_kwargs.update(
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
            device_map="auto",
        )
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, **model_kwargs)
    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="SEQ_2_SEQ_LM"),
    )
    model.print_trainable_parameters()
    train = pd.read_parquet(args.data_dir / "train.parquet").sample(n=args.train_examples, random_state=42)
    validation = pd.read_parquet(args.data_dir / "validation.parquet").sample(n=args.validation_examples, random_state=42)
    report_to = "wandb" if args.wandb and os.getenv("WANDB_API_KEY") else "none"
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=4, per_device_eval_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, predict_with_generate=False, report_to=report_to,
        fp16=use_qlora, optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=DialogueDataset(train, tokenizer), eval_dataset=DialogueDataset(validation, tokenizer),
    )
    result = trainer.train()
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    run = {
        "base_model": MODEL_ID, "method": "QLoRA-4bit-NF4" if use_qlora else "LoRA fallback",
        "train_examples": len(train), "validation_examples": len(validation), "epochs": args.epochs,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()), "train_loss": result.training_loss,
    }
    (args.output_dir / "training_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
