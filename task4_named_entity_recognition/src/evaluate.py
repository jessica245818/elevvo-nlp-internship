"""Compare rule-based and model-based NER on CoNLL-2003."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
from spacy import displacy
from spacy.tokens import Doc, Span


TAG_NAMES = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}
SPACY_TO_CONLL = {
    "PERSON": "PER",
    "PER": "PER",
    "ORG": "ORG",
    "MISC": "MISC",
    "GPE": "LOC",
    "LOC": "LOC",
    "FAC": "LOC",
    "NORP": "MISC",
    "EVENT": "MISC",
    "PRODUCT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC",
}
DISPLAY_COLORS = {
    "PER": "#ffd6a5",
    "ORG": "#caffbf",
    "LOC": "#9bf6ff",
    "MISC": "#bdb2ff",
}


def gold_spans(tags: Iterable[int]) -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    start: int | None = None
    active_label: str | None = None
    tag_list = list(tags)
    for index, tag_id in enumerate(tag_list + [0]):
        tag = TAG_NAMES[int(tag_id)]
        prefix, _, label = tag.partition("-")
        if prefix == "B" or (prefix == "I" and label != active_label) or prefix == "O":
            if start is not None and active_label is not None:
                spans.add((start, index, active_label))
            start = index if prefix in {"B", "I"} else None
            active_label = label if prefix in {"B", "I"} else None
    return spans


def build_gazetteer_patterns(train: pd.DataFrame) -> list[dict[str, object]]:
    label_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for tokens, tags in zip(train["tokens"], train["ner_tags"]):
        for start, end, label in gold_spans(tags):
            label_counts[tuple(tokens[start:end])][label] += 1

    patterns: list[dict[str, object]] = []
    for entity_tokens, counts in label_counts.items():
        label = counts.most_common(1)[0][0]
        patterns.append(
            {
                "label": label,
                "pattern": [{"ORTH": token} for token in entity_tokens],
            }
        )
    return patterns


def make_rule_pipeline(train: pd.DataFrame) -> spacy.language.Language:
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
    patterns = build_gazetteer_patterns(train)
    ruler.add_patterns(patterns)
    print(f"Built rule-based gazetteer with {len(patterns):,} patterns")
    return nlp


def predicted_spans(doc: Doc) -> set[tuple[int, int, str]]:
    return {
        (entity.start, entity.end, mapped_label)
        for entity in doc.ents
        if (mapped_label := SPACY_TO_CONLL.get(entity.label_)) is not None
    }


def score_pipeline(
    name: str,
    nlp: spacy.language.Language,
    test: pd.DataFrame,
    sample_count: int = 12,
) -> tuple[dict[str, object], list[Doc]]:
    docs = (
        Doc(
            nlp.vocab,
            words=list(tokens),
            spaces=[True] * (len(tokens) - 1) + [False],
        )
        for tokens in test["tokens"]
    )
    true_positive = 0
    predicted_total = 0
    gold_total = 0
    per_label = defaultdict(lambda: {"tp": 0, "predicted": 0, "gold": 0})
    samples: list[Doc] = []

    for row_index, doc in enumerate(nlp.pipe(docs, batch_size=64)):
        gold = gold_spans(test.iloc[row_index]["ner_tags"])
        predicted = predicted_spans(doc)
        matches = gold.intersection(predicted)
        true_positive += len(matches)
        predicted_total += len(predicted)
        gold_total += len(gold)
        for _, _, label in gold:
            per_label[label]["gold"] += 1
        for _, _, label in predicted:
            per_label[label]["predicted"] += 1
        for _, _, label in matches:
            per_label[label]["tp"] += 1
        if len(samples) < sample_count and doc.ents:
            samples.append(doc)

    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    label_metrics = {}
    for label in ("PER", "ORG", "LOC", "MISC"):
        counts = per_label[label]
        label_precision = counts["tp"] / counts["predicted"] if counts["predicted"] else 0.0
        label_recall = counts["tp"] / counts["gold"] if counts["gold"] else 0.0
        label_f1 = (
            2 * label_precision * label_recall / (label_precision + label_recall)
            if label_precision + label_recall
            else 0.0
        )
        label_metrics[label] = {
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
        }
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_spans": true_positive,
        "predicted_spans": predicted_total,
        "gold_spans": gold_total,
        "per_label": label_metrics,
    }
    print(f"{name}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    return metrics, samples


def save_entity_examples(samples: list[Doc], model_name: str, output_dir: Path) -> None:
    rows = []
    for sentence_index, doc in enumerate(samples, start=1):
        for entity in doc.ents:
            label = SPACY_TO_CONLL.get(entity.label_)
            if label is not None:
                rows.append(
                    {
                        "sentence_id": sentence_index,
                        "model": model_name,
                        "sentence": doc.text,
                        "entity": entity.text,
                        "category": label,
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "entity_examples.csv", index=False)


def normalize_for_display(doc: Doc) -> Doc:
    display_doc = Doc(
        doc.vocab,
        words=[token.text for token in doc],
        spaces=[bool(token.whitespace_) for token in doc],
    )
    display_doc.ents = [
        Span(display_doc, entity.start, entity.end, label=mapped_label)
        for entity in doc.ents
        if (mapped_label := SPACY_TO_CONLL.get(entity.label_)) is not None
    ]
    return display_doc


def save_displacy(samples: list[Doc], filename: str, output_dir: Path) -> None:
    display_docs = [normalize_for_display(doc) for doc in samples]
    html = displacy.render(
        display_docs,
        style="ent",
        page=True,
        options={"ents": list(DISPLAY_COLORS), "colors": DISPLAY_COLORS},
    )
    (output_dir / filename).write_text(html, encoding="utf-8")


def plot_comparison(results: dict[str, dict[str, object]], output_dir: Path) -> None:
    rows = [
        {"approach": approach, "metric": metric, "score": metrics[metric]}
        for approach, metrics in results.items()
        for metric in ("precision", "recall", "f1")
    ]
    plt.figure(figsize=(10, 5))
    sns.barplot(data=pd.DataFrame(rows), x="metric", y="score", hue="approach")
    plt.ylim(0, 1)
    plt.title("CoNLL-2003 Exact-Span NER Comparison")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(output_dir / "ner_approach_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("task4_named_entity_recognition/data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task4_named_entity_recognition/artifacts"),
    )
    parser.add_argument("--max-test-sentences", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.data_dir / "train.parquet", columns=["tokens", "ner_tags"])
    test = pd.read_parquet(args.data_dir / "test.parquet", columns=["tokens", "ner_tags"])
    if args.max_test_sentences is not None:
        test = test.head(args.max_test_sentences)
    print(f"Evaluating {len(test):,} CoNLL-2003 test sentences")

    pipelines = {
        "rule_based_gazetteer": make_rule_pipeline(train),
        "en_core_web_sm": spacy.load("en_core_web_sm"),
        "en_core_web_md": spacy.load("en_core_web_md"),
    }
    results: dict[str, dict[str, object]] = {}
    samples_by_model: dict[str, list[Doc]] = {}
    for name, pipeline in pipelines.items():
        results[name], samples_by_model[name] = score_pipeline(name, pipeline, test)

    best_model = max(results, key=lambda name: float(results[name]["f1"]))
    save_entity_examples(samples_by_model[best_model], best_model, args.output_dir)
    save_displacy(samples_by_model[best_model], "best_model_entities.html", args.output_dir)
    save_displacy(samples_by_model["rule_based_gazetteer"], "rule_based_entities.html", args.output_dir)
    plot_comparison(results, args.output_dir)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Best approach: {best_model}")
    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
