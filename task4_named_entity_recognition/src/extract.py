"""Extract and categorize entities from user-provided news text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import spacy
from spacy import displacy

from task4_named_entity_recognition.src.evaluate import (
    DISPLAY_COLORS,
    SPACY_TO_CONLL,
    normalize_for_display,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="News article text")
    parser.add_argument("--model", default="en_core_web_md")
    parser.add_argument("--html", type=Path, help="Optional highlighted HTML output")
    args = parser.parse_args()

    nlp = spacy.load(args.model)
    doc = nlp(args.text)
    entities = [
        {"text": entity.text, "category": category}
        for entity in doc.ents
        if (category := SPACY_TO_CONLL.get(entity.label_)) is not None
    ]
    print(json.dumps(entities, indent=2))

    if args.html:
        html = displacy.render(
            normalize_for_display(doc),
            style="ent",
            page=True,
            options={"ents": list(DISPLAY_COLORS), "colors": DISPLAY_COLORS},
        )
        args.html.write_text(html, encoding="utf-8")
        print(f"Saved highlighted entities to {args.html}")


if __name__ == "__main__":
    main()
