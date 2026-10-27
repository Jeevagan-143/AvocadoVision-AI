
from pathlib import Path
import json
import csv

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"

sources = [
    ("RandomForest", REPORT / "classical_results.json"),
    ("MobileNetV2", REPORT / "mobilenetv2_results.json"),
]

rows = []

for name, path in sources:
    if not path.exists():
        print(f"Missing: {path}")
        continue

    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        data = next(
            (item for item in data if item.get("split") == "test"),
            data[0]
        )

    rows.append({
        "model": name,
        "test_accuracy": data["accuracy"],
        "macro_precision": data["precision_macro"],
        "macro_recall": data["recall_macro"],
        "macro_f1": data["f1_macro"],
    })

if not rows:
    raise RuntimeError("No model result files found.")

output = REPORT / "model_comparison.csv"

with output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nMODEL COMPARISON")
for row in rows:
    print(
        f'{row["model"]}: '
        f'Accuracy={row["test_accuracy"]:.2%}, '
        f'Macro F1={row["macro_f1"]:.2%}'
    )

print(f"\nSaved: {output}")
