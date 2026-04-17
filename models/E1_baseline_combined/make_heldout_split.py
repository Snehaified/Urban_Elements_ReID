"""
Create held-out validation split.
- Training pool: UAMEXTERNAL train + Urban2026 train
- Val: UAMEXTERNAL held-out IDs only (matches UAM query protocol for local evaluation)
- Singletons (1-image IDs) are dropped from training.
"""
import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
VAL_FRACTION = 0.15
MIN_IMAGES_PER_ID = 2  # drop singletons

UAM_ROOT = Path("./UAMEXTERNAL")
URBAN_ROOT = Path("./Urban2026")
OUT_DIR = Path("./splits")
OUT_DIR.mkdir(exist_ok=True)

def normalize_class(c):
    c = str(c).strip().lower()
    mapping = {
        "trafficsignal": "trafficsign", "trafficsign": "trafficsign",
        "container": "container",
        "rubbishbins": "rubbishbin", "rubbishbin": "rubbishbin",
        "crosswalk": "crosswalk", "crosswalks": "crosswalk",
    }
    return mapping.get(c, c)

# --- Load UAM train ---
uam = pd.read_csv(UAM_ROOT / "train_classes.csv")
uam["class_norm"] = uam["Class"].apply(normalize_class)
uam["source"] = "uam"
uam["img_path"] = uam["imageName"].apply(lambda x: str(UAM_ROOT / "image_train" / x))
uam["objectID"] = uam["objectID"].apply(lambda x: f"uam_{x}")

# --- Load Urban2026 train ---
urban = pd.read_csv(URBAN_ROOT / "train_classes.csv")
urban["class_norm"] = urban["Class"].apply(normalize_class)
urban["source"] = "urban"
urban["img_path"] = urban["imageName"].apply(lambda x: str(URBAN_ROOT / "image_train" / x))
urban["objectID"] = urban["Corresponding Indexes"].apply(lambda x: f"urban_{x}")

# --- Filter singletons ---
for df, name in [(uam, "UAM"), (urban, "Urban")]:
    counts = df.groupby("objectID").size()
    singletons = counts[counts < MIN_IMAGES_PER_ID].index
    before = len(df)
    df.drop(df[df["objectID"].isin(singletons)].index, inplace=True)
    print(f"  {name}: dropped {before - len(df)} singleton images ({len(singletons)} IDs)")

# --- Pick val IDs from UAM only, requiring c004 + another cam ---
rng = np.random.RandomState(SEED)
val_ids = []

for cls in uam["class_norm"].unique():
    cls_df = uam[uam["class_norm"] == cls]
    id_cams = cls_df.groupby("objectID")["cameraID"].apply(set)
    eligible = [oid for oid, cams in id_cams.items()
                if "c004" in cams and len(cams - {"c004"}) >= 1]
    n_val = max(5, int(len(eligible) * VAL_FRACTION))
    if len(eligible) < n_val:
        n_val = len(eligible)
    if n_val == 0:
        print(f"[WARN] {cls}: no eligible IDs")
        continue
    selected = rng.choice(eligible, size=n_val, replace=False)
    val_ids.extend(selected)
    print(f"  {cls}: {len(eligible)} eligible UAM IDs, held out {n_val}")

val_ids = set(val_ids)

# --- Build splits ---
uam_train = uam[~uam["objectID"].isin(val_ids)]
uam_val = uam[uam["objectID"].isin(val_ids)]

cols = ["cameraID", "imageName", "objectID", "class_norm", "source", "img_path"]
train_df = pd.concat([uam_train[cols], urban[cols]], ignore_index=True)

val_query = uam_val[uam_val["cameraID"] == "c004"].copy()
val_gallery = uam_val[uam_val["cameraID"] != "c004"].copy()

print(f"\n=== Final splits ===")
print(f"  train:        {len(train_df):6d} images, {train_df['objectID'].nunique():5d} IDs")
print(f"    UAM:        {len(uam_train):6d} images, {uam_train['objectID'].nunique():5d} IDs")
print(f"    Urban:      {len(urban):6d} images, {urban['objectID'].nunique():5d} IDs")
print(f"  val_query:    {len(val_query):6d} images, {val_query['objectID'].nunique():5d} IDs")
print(f"  val_gallery:  {len(val_gallery):6d} images, {val_gallery['objectID'].nunique():5d} IDs")

print(f"\nPer-class train IDs:")
print(train_df.groupby(["class_norm", "source"])["objectID"].nunique().unstack(fill_value=0).to_string())

print(f"\nPer-class val query counts:")
print(val_query["class_norm"].value_counts().to_string())

# Checks
assert len(set(uam_train["objectID"]) & val_ids) == 0, "Val leakage!"
assert set(val_query["objectID"]).issubset(set(val_gallery["objectID"])), \
    "Val queries without gallery match"

train_df.to_csv(OUT_DIR / "train.csv", index=False)
val_query.to_csv(OUT_DIR / "val_query.csv", index=False)
val_gallery.to_csv(OUT_DIR / "val_gallery.csv", index=False)
print(f"\nSaved to {OUT_DIR.resolve()}")