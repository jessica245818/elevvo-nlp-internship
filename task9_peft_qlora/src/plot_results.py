"""Plot the saved base-versus-fine-tuned ROUGE comparison."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


artifact_dir = Path("task9_peft_qlora/artifacts")
metrics = json.loads((artifact_dir / "rouge_comparison.json").read_text())
rows = [
    {"model": model.replace("_", " ").title(), "metric": metric.upper(), "score": score}
    for model in ("base", "fine_tuned")
    for metric, score in metrics[model].items()
]
plt.figure(figsize=(9, 5))
axis = sns.barplot(data=pd.DataFrame(rows), x="metric", y="score", hue="model")
axis.set_ylim(0, 55)
axis.set_xlabel("")
axis.set_ylabel("ROUGE F1")
axis.set_title("SAMSum PEFT Fine-Tuning Comparison")
for container in axis.containers:
    axis.bar_label(container, fmt="%.1f")
plt.tight_layout()
plt.savefig(artifact_dir / "rouge_comparison.png", dpi=180)
plt.close()
