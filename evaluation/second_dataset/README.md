# SECOND (image source for CDVQA — license unstated, user-approved fetch)

- **Source**: official project page — [captain-whu.github.io/SCD](https://captain-whu.github.io/SCD/)
  (Ji, Wei, Lu et al., semantic change detection dataset, Wuhan University).
- **License**: **not stated anywhere on the official page.** This is the
  one dataset in this build fetched under an explicitly *unstated*
  license rather than a clean permissive one (compare RSVQA-LR's
  CC-BY-4.0 or CDVQA's Apache-2.0) — flagged to the user with the real
  size and this exact caveat before fetching; used here strictly for
  internal, non-commercial evaluation of an already-built specialist, not
  redistributed or shipped as part of the product.
- **Distribution**: Google Drive only, no HuggingFace/Kaggle mirror
  found. Download IS scriptable (tested, works) despite Google Drive's
  large-file "virus scan warning" interstitial — see the command below.
- **Size**: the public release zip is **~3.79GB** (verified via a real
  `content-length` header on the actual download, not an estimate).
- **What's used**: only the 968 image pairs CDVQA's test split actually
  references (out of the ~2,968 pairs in the public release) — the full
  zip must still be fetched since it's one archive, but nothing beyond
  those 968 pairs is read by `runners/cdvqa_adapter.py`.

## Getting the raw data (not committed — see `.gitignore`)

```bash
mkdir -p evaluation/second_dataset/raw
cd evaluation/second_dataset/raw
FILE_ID="1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u"
curl -sL -c cookies.txt -o scan_warning.html \
  "https://drive.google.com/uc?export=download&id=${FILE_ID}"
UUID=$(grep -oE 'name="uuid" value="[^"]*"' scan_warning.html | sed -E 's/.*value="([^"]*)"/\1/')
curl -sL -b cookies.txt -C - -o second_dataset.zip \
  "https://drive.usercontent.google.com/download?id=${FILE_ID}&export=download&confirm=t&uuid=${UUID}"
unzip -q second_dataset.zip
# The zip contains a nested RAR archive (real, verified) — needs a RAR-aware
# tool; macOS has none built in:
brew install unar
unar -o . SECOND_train_set.rar
```

The two-step curl (fetch the warning page, extract its `uuid` form
field, retry with `confirm=t&uuid=...`) is Google Drive's real
large-file bypass — a plain single `curl` on the share link returns an
HTML interstitial, not the file, for anything over ~100MB. Use `-C -`
(resume) and be ready to retry: this download hit a real, silent
mid-transfer truncation once during this project's own fetch (curl's
`--retry-all-errors` combined with `-C -` misbehaved on a connection
hiccup and quietly lost several hundred MB without erroring) — verify
the final file with `unzip -t` before trusting it's actually complete,
don't rely on the reported size alone.

After extraction, the real layout is
`SECOND_train_set/{im1,im2,label1,label2}/<id>.png` (2,968 real pairs —
CDVQA's 968-pair test split is a subset of these, referenced by the same
`<id>.png` filenames) — verified against the real archive, not assumed.
`runners/cdvqa_adapter.py`'s `SECOND_IMAGES_DIR` points here already; if
a future re-download changes the layout, fix that constant rather than
silently renaming folders to match old assumptions.
