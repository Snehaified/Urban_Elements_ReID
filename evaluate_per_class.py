import os
import numpy as np
import pandas as pd
import argparse


def read_csv_gt(csv_path):
    """Read Ground Truth (test.csv or query.csv)"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip()
    col_id = next((c for c in ['objectID', 'Corresponding Indexes', 'object'] if c in df.columns), None)
    if col_id is None:
        raise ValueError(f"ID column not found in {csv_path}")
    dataset = {}
    for _, row in df.iterrows():
        dataset[row['imageName']] = int(row[col_id])
    return dataset, df['imageName'].tolist()


def read_prediction_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    col_pred = next((c for c in ['Corresponding Indexes', 'predictions'] if c in df.columns), None)
    if col_pred is None:
        raise ValueError("Prediction CSV must have a 'Corresponding Indexes' column")
    predictions = {}
    for _, row in df.iterrows():
        indices = [int(i) for i in str(row[col_pred]).split()]
        predictions[row['imageName']] = indices
    return predictions


def read_class_csv(csv_path):
    """Read class labels: imageName -> class (lowercased, stripped)"""
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip()
    return {row['imageName']: row['Class'].lower().strip() for _, row in df.iterrows()}


# --- CONFIGURATION ---
parser = argparse.ArgumentParser(description="ReID Per-Class Eval")
parser.add_argument("--track", default="submission.csv", help="CSV with predictions")
parser.add_argument("--path", default="./Urban2026/", help="Folder with CSVs")
args = parser.parse_args()

# --- Load data ---
gallery_dict, gallery_names = read_csv_gt(os.path.join(args.path, 'test.csv'))
sorted_gallery_names = sorted(gallery_names, key=lambda x: int(x.split('.')[0]))
id_gallery = np.array([gallery_dict[name] for name in sorted_gallery_names])

query_dict, query_names = read_csv_gt(os.path.join(args.path, 'query.csv'))
preds_dict = read_prediction_csv(args.track)

query_class = read_class_csv(os.path.join(args.path, 'query_classes.csv'))

# --- Per-class accumulators ---
per_class_ap = {}    # class -> list of APs
per_class_cmc = {}   # class -> list of CMC arrays
per_class_count = {}

sample_key = next(iter(preds_dict))
ranking_len = len(preds_dict[sample_key])

print(f"Evaluating {len(query_names)} queries...")

for q_name in query_names:
    if q_name not in preds_dict:
        continue

    query_id = query_dict[q_name]
    q_cls = query_class.get(q_name, "unknown")
    pred_indices = np.array(preds_dict[q_name]) - 1
    sortID = id_gallery[pred_indices]

    true_positives_in_gallery = np.where(id_gallery == query_id)[0]
    if len(true_positives_in_gallery) == 0:
        continue

    rows_good = np.where(sortID == query_id)[0]

    ap = 0.0
    cmc = np.zeros(ranking_len)
    ngood = len(true_positives_in_gallery)

    if rows_good.size != 0:
        cmc[rows_good[0]:] = 1
        for i, pos in enumerate(rows_good):
            precision = (i + 1) / (pos + 1)
            if pos != 0:
                old_precision = (i + 1) / (pos + 1)
            else:
                old_precision = 1.0
            ap += (1.0 / ngood) * (old_precision + precision) / 2

    per_class_ap.setdefault(q_cls, []).append(ap)
    per_class_cmc.setdefault(q_cls, []).append(cmc)
    per_class_count[q_cls] = per_class_count.get(q_cls, 0) + 1

# --- Report ---
print("\n" + "=" * 70)
print(f"{'Class':<16} {'N':>5} {'mAP':>10} {'Rank-1':>10} {'Rank-5':>10} {'Rank-10':>10}")
print("-" * 70)

all_ap = []
all_cmc = []
for cls in sorted(per_class_ap.keys()):
    aps = np.array(per_class_ap[cls])
    cmcs = np.stack(per_class_cmc[cls])
    mean_cmc = cmcs.mean(axis=0)
    print(
        f"{cls:<16} {len(aps):>5} "
        f"{aps.mean():>10.6f} "
        f"{mean_cmc[0]:>10.6f} "
        f"{mean_cmc[4]:>10.6f} "
        f"{mean_cmc[9]:>10.6f}"
    )
    all_ap.extend(aps)
    all_cmc.extend(cmcs)

print("-" * 70)
all_ap_arr = np.array(all_ap)
all_cmc_arr = np.stack(all_cmc)
print(
    f"{'OVERALL':<16} {len(all_ap):>5} "
    f"{all_ap_arr.mean():>10.6f} "
    f"{all_cmc_arr.mean(axis=0)[0]:>10.6f} "
    f"{all_cmc_arr.mean(axis=0)[4]:>10.6f} "
    f"{all_cmc_arr.mean(axis=0)[9]:>10.6f}"
)
print("=" * 70)