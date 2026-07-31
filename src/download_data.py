"""Download and extract Stanford's Large Movie Review Dataset."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path


DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
EXPECTED_MD5 = "7c2ac02c03563afcf9b574c7e56c153a"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - used only for dataset integrity verification
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"Unsafe archive path: {member.name}")
    # Paths are validated above; Python 3.9 does not support extractall(filter=...).
    archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.data_dir / "aclImdb_v1.tar.gz"
    extracted_path = args.data_dir / "aclImdb"

    if extracted_path.exists():
        print(f"Dataset already exists at {extracted_path}")
        return

    if not archive_path.exists():
        print(f"Downloading {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, archive_path)

    actual_md5 = md5sum(archive_path)
    if actual_md5 != EXPECTED_MD5:
        raise ValueError(f"Checksum mismatch: expected {EXPECTED_MD5}, got {actual_md5}")

    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, args.data_dir)
    print(f"Dataset ready at {extracted_path}")


if __name__ == "__main__":
    main()
