# Task 9: Efficient LLM Fine-Tuning with PEFT/QLoRA

Resource-efficient specialization of `google/flan-t5-small` for SAMSum dialogue summarization. On a CUDA GPU the training script uses 4-bit NF4 double quantization plus LoRA on every linear layer (QLoRA). On unsupported hardware it automatically uses the identical LoRA configuration without quantization so the pipeline remains testable.

## Single-GPU optimization

- 4-bit NF4 weights with nested/double quantization
- rank-16 LoRA; only adapter parameters are trained
- gradient accumulation and batch size four
- paged 8-bit AdamW on CUDA
- 512-token inputs and 96-token targets
- fits a free Colab T4; no multi-GPU configuration is used

Apple MPS cannot run bitsandbytes 4-bit quantization, so results produced locally are labeled **LoRA fallback**, never QLoRA. Run the same command on a T4 to activate QLoRA automatically.

## Measured results

The local single-device LoRA fallback trained on 512 dialogues for two epochs and was evaluated against the unchanged base model on the same fixed 100-dialogue test sample.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Macro average |
| --- | ---: | ---: | ---: | ---: |
| Base FLAN-T5 | **45.92** | 22.05 | 37.66 | 35.21 |
| PEFT fine-tuned | 45.80 | **22.67** | **38.03** | **35.50** |

The adapter improved ROUGE-2 by 0.62, ROUGE-L by 0.38, and macro-average ROUGE by 0.29, while ROUGE-1 decreased by 0.11. This is a modest overall win from training 2.56 million parameters (3.21% of the model), and the tradeoff is reported rather than hidden. The CUDA/T4 QLoRA configuration should be rerun to produce hardware-specific 4-bit results.

## Run

```bash
source .venv/bin/activate
pip install -r task9_peft_qlora/requirements.txt
python -m task9_peft_qlora.src.download_data
python -m task9_peft_qlora.src.train
python -m task9_peft_qlora.src.evaluate
```

Use `--wandb` with `WANDB_API_KEY` to log training. For a quick smoke test, set `--train-examples 64 --validation-examples 32 --epochs 1`.

## Bonus: merge or publish

```bash
python -m task9_peft_qlora.src.merge_adapter
python -m task9_peft_qlora.src.push_to_hub your-username/flan-t5-samsum-qlora
```

Publishing requires prior `huggingface-cli login`; it is intentionally not performed automatically.

## Dataset

[SAMSum](https://huggingface.co/datasets/Samsung/samsum) contains messenger-style dialogues paired with human summaries. The downloader uses a data-only Parquet mirror and validates the train, validation, and test splits.

## Outputs

- `training_run.json` — method, resource use, trainable parameters, and loss
- `rouge_comparison.json` — base versus fine-tuned ROUGE-1/2/L
- `summary_comparison.csv` — qualitative before/after examples
- `rouge_comparison.png` — base versus adapted score chart
- `adapter/` — compact LoRA adapter (excluded from Git)
