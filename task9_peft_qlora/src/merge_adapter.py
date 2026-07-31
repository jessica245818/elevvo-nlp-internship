"""Merge a trained LoRA adapter into the base model."""

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from task9_peft_qlora.src.train import MODEL_ID


adapter = Path("task9_peft_qlora/artifacts/adapter")
output = Path("task9_peft_qlora/artifacts/merged_model")
model = PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID), adapter).merge_and_unload()
model.save_pretrained(output, safe_serialization=True)
AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(output)
print(f"Merged standalone model saved to {output}")
