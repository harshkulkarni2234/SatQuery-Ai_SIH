# Scenario C — optical + SAR pair (cross-modal demo)

`optical.png` (Sentinel-2, RGB, 120x120) / `sar.png` (Sentinel-1, grayscale,
120x120) — a genuinely paired optical + SAR patch:

- `optical.png` = `S2B_MSIL2A_20170808T094029_N9999_R036_T35ULA_62_00`
- `sar.png` = `S1A_IW_GRDH_1SDV_20170808T043459_35ULA_62_00`

Both filenames encode the same MGRS tile (`T35ULA`) and the same patch grid
index (`62_00`), and both were acquired on the same real-world date
(2017-08-08, ~5 hours apart — Sentinel-1 at 04:34 UTC, Sentinel-2 at 09:40
UTC). This is exactly BigEarthNet-MM's own patch-pairing convention: patches
sharing a tile ID and grid index are pre-aligned to the same pixel grid by
construction, which is what "genuinely co-registered" means here.

## Known limitation — SatQuery will report this pair as "unverified", correctly

These two patches ship as plain PNG, which carries no embedded CRS or
geotransform. SatQuery's `extract_metadata()` correctly reports
`is_georeferenced: false` and `crs: null` for both files as delivered, and
its `compatibility.check_optical_sar_pair()` will therefore report
`coregistration: "unverified"` — **not** `"verified"` — even though the two
patches are, in fact, co-registered by BigEarthNet-MM's own construction.
This is intentional, correct, honest behavior: the system only claims
"verified" when it can check matching CRS + pixel transforms from the files
themselves (see [docs/CONTRACTS.md](../../../docs/CONTRACTS.md)), and these
files don't carry that metadata. Don't "fix" this by fabricating a
geotransform for these files — if exact per-patch coordinates are ever
needed, they'd have to come from the original BigEarthNet-MM product's
`MTD_TL.xml` tile metadata, which was not available with this download.

## Source and licence

Filename convention and patch pairing match the
[BigEarthNet-MM](https://bigearth.net/) dataset (Sentinel-1/Sentinel-2,
licensed CC-BY 4.0 by TU Berlin's RSiM group / BigEarthNet). These two files
were found already-extracted in a local `testing/` folder with no
accompanying manifest, label file, or licence text, so **this citation is
inferred from the filenames, not independently confirmed** — if the original
download source is found, update this with the exact citation, DOI, and
licence text per BigEarthNet-MM's terms.

The remaining ~4,000 single-modality Sentinel-2 patches also found in the
`testing/` folder (from two different tiles/dates, `T34UDG`/2017-07-20 and
`T35ULA`/2017-08-08) are not used here — they're optical-only, from
different scenes, and unaccompanied by the labels file BigEarthNet's own
Q/A prep (Phase B2) would need. They're left in place for a future phase
to pick up if the corresponding label file can be sourced.
