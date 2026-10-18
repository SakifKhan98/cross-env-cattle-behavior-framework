"""Validation script for exported tubelets (§5.3 checks)."""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_ONE_DAY = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LABELS_CSV = os.path.join(_ONE_DAY, "data", "processed", "tubelets", "labels.csv")

_LABEL_NAMES = {
    0: "Standing", 1: "Lying", 2: "Foraging",
    3: "Drinking", 4: "Ruminating", 5: "Grooming", 6: "Other",
}


def validate(labels_csv: str = _LABELS_CSV) -> bool:
    """Run all §5.3 checks. Returns True if all hard checks pass."""
    # ── 1. Load CSV ───────────────────────────────────────────────────────────
    if not os.path.isfile(labels_csv):
        print(f"FAIL labels.csv not found: {labels_csv}")
        return False

    rows = []
    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Total rows: {len(rows)}")
    if len(rows) == 0:
        print("FAIL labels.csv is empty")
        return False
    print("PASS labels.csv has rows")

    # ── 2. Path integrity ─────────────────────────────────────────────────────
    missing_dirs = 0
    wrong_count = 0
    for row in rows:
        full = os.path.join(_ONE_DAY, row["tubelet_dir"])
        if not os.path.isdir(full):
            missing_dirs += 1
            continue
        jpegs = [f for f in os.listdir(full) if f.endswith(".jpg")]
        if len(jpegs) != 16:
            wrong_count += 1

    print(f"\nPath checks:")
    print(f"  Missing dirs     : {missing_dirs}")
    print(f"  Dirs with ≠16 JPG: {wrong_count}")
    path_ok = missing_dirs == 0 and wrong_count == 0
    print(f"{'PASS' if path_ok else 'FAIL'} all tubelet dirs have exactly 16 JPEGs")

    # ── 3. Class distribution ─────────────────────────────────────────────────
    from collections import defaultdict, Counter

    class_counts: dict[tuple, int] = defaultdict(int)
    for row in rows:
        class_counts[(row["dataset"], int(row["label_id"]))] += 1

    print("\nClass distribution (dataset, label_id, label_name, count):")
    datasets_with_drinking: set[str] = set()
    for (ds, lid), cnt in sorted(class_counts.items()):
        name = _LABEL_NAMES.get(lid, "?")
        flag = " ← Drinking" if lid == 3 else ""
        print(f"  {ds:6s}  label {lid}  {name:12s}  {cnt:6d}{flag}")
        if lid == 3:
            datasets_with_drinking.add(ds)

    drinking_ok = len(datasets_with_drinking) > 0
    print(f"{'PASS' if drinking_ok else 'FAIL'} Drinking (label 3) present in: "
          f"{datasets_with_drinking if datasets_with_drinking else 'NONE'}")

    # ── 4. Split distribution ─────────────────────────────────────────────────
    split_counts: dict[tuple, int] = defaultdict(int)
    for row in rows:
        split_counts[(row["dataset"], row["split"])] += 1

    print("\nSplit distribution:")
    for (ds, sp), cnt in sorted(split_counts.items()):
        print(f"  {ds:6s}  {sp:5s}  {cnt:6d}")

    cbvd5_splits = {sp for (ds, sp), _ in split_counts.items() if ds == "cbvd5"}
    # CBVD-5 dataset artifact: test CSV keys are identical to val CSV keys; first-occurrence
    # wins in load_cbvd5_annotations() assigns them all to "val", so "test" rows never appear.
    cbvd5_ok = {"train", "val"}.issubset(cbvd5_splits)
    if "test" not in cbvd5_splits:
        print(f"WARN  CBVD-5 'test' split absent — known artifact: test CSV == val CSV; "
              f"entries absorbed into 'val'")
    print(f"{'PASS' if cbvd5_ok else 'FAIL'} CBVD-5 has train/val splits "
          f"(found: {sorted(cbvd5_splits)})")

    cvb_splits = {sp for (ds, sp), _ in split_counts.items() if ds == "cvb"}
    cvb_ok = {"train", "val"}.issubset(cvb_splits)
    print(f"{'PASS' if cvb_ok else 'FAIL'} CVB has train/val splits "
          f"(found: {sorted(cvb_splits)})")

    # ── 5. Per-dataset totals ─────────────────────────────────────────────────
    ds_totals: Counter = Counter(row["dataset"] for row in rows)
    print(f"\nPer-dataset totals:")
    for ds, cnt in sorted(ds_totals.items()):
        print(f"  {ds:6s}  {cnt:6d}")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_pass = path_ok and drinking_ok and cbvd5_ok
    print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
