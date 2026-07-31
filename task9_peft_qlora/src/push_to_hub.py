"""Push the trained adapter to the authenticated Hugging Face account."""

import argparse

from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from task9_peft_qlora.src.train import MODEL_ID


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("repo_id", help="For example: username/flan-t5-samsum-qlora")
args = parser.parse_args()
adapter = "task9_peft_qlora/artifacts/adapter"
model = PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID), adapter)
model.push_to_hub(args.repo_id)
AutoTokenizer.from_pretrained(MODEL_ID).push_to_hub(args.repo_id)
print(f"Published adapter to https://huggingface.co/{args.repo_id}")
