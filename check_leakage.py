import pandas as pd
from pathlib import Path

UAM = Path("./UAMEXTERNAL")
URBAN = Path("./Urban2026")

uam_train = pd.read_csv(UAM / "train_classes.csv")
uam_query = pd.read_csv(UAM / "query_classes.csv")
uam_test = pd.read_csv(UAM / "test_classes.csv")
urban_train = pd.read_csv(URBAN / "train_classes.csv")

print("=" * 60)
print("UAM — identity-level disjointness check")
print("=" * 60)

train_ids = set(uam_train["objectID"])
query_ids = set(uam_query["objectID"])
test_ids = set(uam_test["objectID"])

print(f"  UAM train IDs:     {len(train_ids)}")
print(f"  UAM query IDs:     {len(query_ids)}")
print(f"  UAM test IDs:      {len(test_ids)}")
print()
print(f"  train ∩ query:     {len(train_ids & query_ids)}  <-- must be 0 for no leakage")
print(f"  train ∩ test:      {len(train_ids & test_ids)}   <-- must be 0 for no leakage")
print(f"  query ∩ test:      {len(query_ids & test_ids)}   <-- should equal query size (107+46=153)")

print()
print("=" * 60)
print("UAM — image-level disjointness check")
print("=" * 60)

# Build (cam, filename) tuples — same filename across splits is ambiguous without cam
train_imgs = set(zip(uam_train["cameraID"], uam_train["imageName"]))
query_imgs = set(zip(uam_query["cameraID"], uam_query["imageName"]))
test_imgs = set(zip(uam_test["cameraID"], uam_test["imageName"]))

print(f"  train ∩ query images: {len(train_imgs & query_imgs)}")
print(f"  train ∩ test images:  {len(train_imgs & test_imgs)}")
print(f"  query ∩ test images:  {len(query_imgs & test_imgs)}")

# Check if filename alone is unique
train_names = set(uam_train["imageName"])
query_names = set(uam_query["imageName"])
test_names = set(uam_test["imageName"])
print()
print(f"  Filename overlap (train vs query): {len(train_names & query_names)}")
print(f"  Filename overlap (train vs test):  {len(train_names & test_names)}")
print("    ^ Non-zero here is NORMAL if splits restart numbering at 000001")
print("      What matters is whether the IMAGES (content) are the same, not filenames")

print()
print("=" * 60)
print("Urban2026 train vs UAM eval — cross-dataset leakage check")
print("=" * 60)
# These have different ID spaces, but check class distribution
print(f"  Urban2026 train IDs: {urban_train['Corresponding Indexes'].nunique()}")
print(f"  (Urban2026 IDs are independent pseudo-IDs, cannot leak into UAM eval)")
print(f"  — but visually similar objects across datasets could cause domain 'leakage'.")
print(f"  — this is not leakage in the strict sense, it's just cross-domain training.")

print()
print("=" * 60)
print("VERDICT")
print("=" * 60)
if len(train_ids & query_ids) == 0 and len(train_ids & test_ids) == 0:
    print("  ✓ NO IDENTITY-LEVEL LEAKAGE in UAM. Train IDs are disjoint from eval.")
    print("  ✓ Safe to train on UAM train and evaluate on UAM query+test.")
else:
    print("  ✗ LEAKAGE DETECTED. Train IDs overlap with eval IDs.")
    print("    You must exclude overlapping IDs from training.")
