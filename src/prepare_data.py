
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "train"
OUT = ROOT / "data" / "processed"

CLASSES = ["blackspot", "canker", "fresh", "grenning"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Dataset folder not found: {RAW}")

    # Rebuild processed data without modifying the original dataset.
    if OUT.exists():
        shutil.rmtree(OUT)

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_val = 0
    total_test = 0

    for cls in CLASSES:
        source = RAW / cls
        official_test = RAW / "test" / cls

        if not source.exists():
            raise FileNotFoundError(f"Missing class folder: {source}")

        files = image_files(source)
        if len(files) < 2:
            raise ValueError(f"Not enough images in class: {cls}")

        train_files, val_files = train_test_split(
            files,
            test_size=0.20,
            random_state=42,
            shuffle=True,
        )

        for file in train_files:
            shutil.copy2(file, OUT / "train" / cls / file.name)

        for file in val_files:
            shutil.copy2(file, OUT / "val" / cls / file.name)

        test_files = image_files(official_test) if official_test.exists() else []
        for file in test_files:
            shutil.copy2(file, OUT / "test" / cls / file.name)

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

        print(
            f"{cls:12} | train={len(train_files):4} "
            f"| val={len(val_files):3} | test={len(test_files):3}"
        )

    print("\nDataset preparation complete!")
    print(f"Training images:   {total_train}")
    print(f"Validation images: {total_val}")
    print(f"Official test images: {total_test}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
