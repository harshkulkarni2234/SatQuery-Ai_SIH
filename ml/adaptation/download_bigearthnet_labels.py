"""Download BigEarthNet labels-only files (user-approved, license CDLA-Permissive-1.0).

Downloads from Zenodo:
  v1.0 train.csv.gz / val.csv.gz / test.csv.gz  (record 12687186)
  v2.0 metadata.parquet                          (record 10891137)

All sizes verified from the Zenodo API before starting.
"""

import gzip
import os
import sys
import urllib.request
import json

DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
os.makedirs(DEST_DIR, exist_ok=True)

FILES = [
    ("v1.0", "https://zenodo.org/records/12687186/files/train.csv.gz?download=1",
     "train.csv.gz", 1639745),
    ("v1.0", "https://zenodo.org/records/12687186/files/val.csv.gz?download=1",
     "val.csv.gz", 797678),
    ("v1.0", "https://zenodo.org/records/12687186/files/test.csv.gz?download=1",
     "test.csv.gz", 810561),
    ("v2.0", "https://zenodo.org/records/10891137/files/metadata.parquet?download=1",
     "metadata.parquet", 3616349),
]


def download(url, dest_path, expected_size):
    if os.path.isfile(dest_path):
        actual = os.path.getsize(dest_path)
        if actual == expected_size:
            print(f"SKIP (already present, size ok): {dest_path}")
            return True
        print(f"RE-download (size mismatch {actual} != {expected_size}): {dest_path}")

    req = urllib.request.Request(url, headers={"User-Agent": "SatQueryAI/1.0"})
    print(f"Downloading {os.path.basename(dest_path)} ...")
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest_path, "wb") as f:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            f.write(chunk)

    actual = os.path.getsize(dest_path)
    if actual != expected_size:
        print(f"ERROR: size mismatch for {dest_path}: expected {expected_size}, got {actual}")
        return False
    print(f"OK {dest_path} ({actual} bytes)")
    return True


def report_overlap():
    """Report how many local testing/ patch bases appear in each label file."""
    testing_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "testing")
    local_names = sorted({
        f[:-4] for f in os.listdir(testing_dir)
        if f.endswith(".png") and not f.startswith(".") and not f.startswith("_")
    })
    print(f"\nLocal testing/ patches: {len(local_names)} base names")

    label_sources = {
        "v1.0": {"train.csv", "val.csv", "test.csv"},
    }
    for version, fnames in label_sources.items():
        all_names = set()
        for fn in fnames:
            gz = os.path.join(DEST_DIR, fn + ".gz")
            if not os.path.isfile(gz):
                continue
            with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as f:
                header = f.readline()
                # Find the patch-name column by inspecting header tokens
                cols = [c.strip() for c in header.rstrip("\n").split(",")]
                print(f"  {fn}: {len(cols)} columns, first row preview:")
                names = []
                counter = 0
                for line in f:
                    if counter > 2:
                        break
                    print(f"    {line.rstrip()[:220]}")
                    counter += 1
                # now full pass
                f.seek(0)
                f.readline()
                for line in f:
                    parts = [p.strip() for p in line.rstrip("\n").split(",")]
                    if parts:
                        key = parts[0].replace('"', '')
                        if key and not key.startswith("S1"):
                            all_names.add(key)
        overlap = all_names & local_names
        print(f"  {version}: {len(all_names)} named patches in label files, "
              f"{len(overlap)} overlap with local testing/")
        if overlap:
            print(f"    sample overlapping names: {sorted(overlap)[:5]}")


def main():
    ok = True
    for version, url, fname, size in FILES:
        ok = download(url, os.path.join(DEST_DIR, fname), size) and ok
    if not ok:
        print("Some downloads failed.")
        sys.exit(1)
    report_overlap()


if __name__ == "__main__":
    main()