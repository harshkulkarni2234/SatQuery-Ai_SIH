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
curl -sL -b cookies.txt -o second_dataset.zip \
  "https://drive.usercontent.google.com/download?id=${FILE_ID}&export=download&confirm=t&uuid=${UUID}"
unzip -q second_dataset.zip
```

The two-step curl (fetch the warning page, extract its `uuid` form
field, retry with `confirm=t&uuid=...`) is Google Drive's real
large-file bypass — a plain single `curl` on the share link returns an
HTML interstitial, not the file, for anything over ~100MB.

After unzipping, this adapter expects `im1/` and `im2/` subfolders of
same-named `.png` files (pre-event / post-event) — the dataset's
documented convention. If the real archive layout differs, fix
`runners/cdvqa_adapter.py`'s `SECOND_IMAGES_DIR` path construction rather
than silently renaming folders to match old assumptions.
