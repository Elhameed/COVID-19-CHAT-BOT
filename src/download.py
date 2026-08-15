"""Fetch the COUGH dataset into ``data/raw/``.

COUGH (Zhang et al.) is a COVID-19 FAQ *retrieval* benchmark. Unlike a bare FAQ
dump it ships relevance judgments, which is what lets us report honest metrics
(PRD §6.2, §8).

The raw files are never committed: the corpus is CC BY-NC-SA 4.0, research and
education use only, and redistribution would carry ShareAlike obligations we
don't want to impose on this repository. This module re-fetches them instead.

Usage::

    python -m src.download            # fetch anything missing
    python -m src.download --force    # re-fetch everything
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Pinned to a commit-less branch ref: COUGH is archival and has not changed
# since publication. If it ever does, the row-count assertions below fail loudly
# rather than silently shifting every metric we report.
BASE_URL = "https://raw.githubusercontent.com/sunlab-osu/covid-faq/master/data"
META_BASE_URL = "https://raw.githubusercontent.com/sunlab-osu/covid-faq/master"


@dataclass(frozen=True)
class RemoteFile:
    """A file to fetch, with the row count we expect it to have."""

    name: str
    base: str
    expected_rows: int | None
    role: str


FILES: tuple[RemoteFile, ...] = (
    RemoteFile(
        "FAQ_Bank_eval.csv",
        BASE_URL,
        7117,
        "retrieval corpus + served knowledge base",
    ),
    RemoteFile("User_Query_Bank.csv", BASE_URL, 1201, "user queries (split train/dev/test)"),
    RemoteFile("Annotated_Relevance_Set.csv", BASE_URL, 39760, "relevance judgments (qrels)"),
    RemoteFile("FAQ_Bank.csv", BASE_URL, 15919, "full multilingual FAQ pool (reference only)"),
    # Not data, but required to comply with the licence: attribution is owed to
    # the sites listed in List_of_websites.txt.
    RemoteFile("LICENCE.md", META_BASE_URL, None, "licence terms"),
    RemoteFile("List_of_websites.txt", META_BASE_URL, None, "attribution list"),
)


def _download(url: str, dest: Path) -> int:
    """Download ``url`` to ``dest``, returning the number of bytes written.

    Writes to a temporary file first so an interrupted run cannot leave a
    truncated CSV that later looks valid.
    """
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} for {url}")
            payload = response.read()
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"could not fetch {url}: {exc}") from exc

    tmp.write_bytes(payload)
    tmp.replace(dest)
    return len(payload)


def _raise_csv_field_limit() -> None:
    """Lift the 128 KB per-field cap that COUGH's longest answers exceed.

    ``sys.maxsize`` overflows the C long the csv module uses on Windows, so step
    down until a value is accepted.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def count_data_rows(path: Path) -> int:
    """Count CSV data rows, excluding the header.

    Uses the csv module rather than counting newlines: COUGH answers contain
    embedded newlines inside quoted fields, so a line count overstates the row
    count by thousands.
    """
    _raise_csv_field_limit()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        return sum(1 for _ in reader)


def fetch(force: bool = False, raw_dir: Path = RAW_DIR) -> dict[str, int]:
    """Fetch all COUGH files. Returns a mapping of filename -> row count."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for spec in FILES:
        dest = raw_dir / spec.name

        if dest.exists() and not force:
            print(f"  = {spec.name:<30} already present, skipping")
        else:
            size = _download(f"{spec.base}/{spec.name}", dest)
            print(f"  + {spec.name:<30} {size / 1024:>9,.0f} KB   {spec.role}")

        if spec.expected_rows is not None:
            rows = count_data_rows(dest)
            counts[spec.name] = rows
            if rows != spec.expected_rows:
                raise RuntimeError(
                    f"{spec.name}: expected {spec.expected_rows:,} data rows, found "
                    f"{rows:,}. The upstream dataset has changed; every metric in this "
                    f"repository is computed against the documented counts, so stop and "
                    f"reconcile before continuing."
                )

    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args(argv)

    print(f"Fetching COUGH into {RAW_DIR.relative_to(PROJECT_ROOT)}/")
    counts = fetch(force=args.force)

    print("\nRow counts verified against PRD §6.2:")
    for name, rows in counts.items():
        print(f"  {name:<30} {rows:>7,}")

    print(
        "\nCOUGH is CC BY-NC-SA 4.0 — research and education use only.\n"
        "Attribution is owed to the sites in List_of_websites.txt."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
