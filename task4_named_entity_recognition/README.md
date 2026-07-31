# Task 4: Named Entity Recognition from News Articles

Rule-based and model-based named entity recognition on CoNLL-2003 news text. The project extracts and categorizes **people**, **organizations**, **locations**, and **miscellaneous entities**, evaluates exact entity spans, and creates highlighted displaCy visualizations.

## Approaches

1. **Rule-based gazetteer:** an exact-token spaCy `EntityRuler` built from entities observed in the CoNLL-2003 training split.
2. **`en_core_web_sm`:** spaCy's small pretrained English pipeline.
3. **`en_core_web_md`:** spaCy's medium pretrained English pipeline with word vectors.

spaCy labels are normalized to CoNLL's `PER`, `ORG`, `LOC`, and `MISC` categories before evaluation.

## Setup and run

From the repository root:

```bash
source .venv/bin/activate
pip install -r task4_named_entity_recognition/requirements.txt
python -m task4_named_entity_recognition.src.download_data
python -m task4_named_entity_recognition.src.evaluate
```

For a quick development run:

```bash
python -m task4_named_entity_recognition.src.evaluate --max-test-sentences 300
```

Extract entities from new article text:

```bash
python -m task4_named_entity_recognition.src.extract \
  "Microsoft opened a new office in Cairo and appointed Jane Smith as director." \
  --html task4_named_entity_recognition/artifacts/custom_entities.html
```

## Evaluation

The evaluator uses strict CoNLL-style exact-span matching. A prediction counts as correct only when its start token, end token, and normalized entity category all match the gold annotation.

### Results on the complete test split

| Approach | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Rule-based gazetteer | 0.7196 | 0.5094 | 0.5965 |
| `en_core_web_sm` | 0.6559 | 0.5450 | 0.5953 |
| **`en_core_web_md`** | **0.6515** | **0.5875** | **0.6178** |

The medium model achieved the best exact-span F1, while the rule-based approach produced the highest precision. This illustrates the gazetteer's conservative behavior and the model-based pipelines' stronger generalization to unseen entities.

## Dataset

The [CoNLL-2003 dataset](https://huggingface.co/datasets/eriktks/conll2003) contains Reuters newswire sentences annotated with person, organization, location, and miscellaneous entities. The downloader uses a data-only Parquet mirror and validates all split sizes and token/tag alignment.

## Outputs

- `metrics.json` with overall and per-category precision, recall, and F1
- `ner_approach_comparison.png`
- `entity_examples.csv`
- `best_model_entities.html` with highlighted model-based entities
- `rule_based_entities.html` with highlighted rule-based entities
