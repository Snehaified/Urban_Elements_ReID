"""
Standalone evaluation on UAM query/test.
Usage:
  python eval_uam.py --config_file config/PAT_combined_E1.yml \
      TEST.WEIGHT ./models/E1_baseline_combined/part_attention_vit_60.pth
"""
import os
import argparse
import csv
import numpy as np
import torch
from collections import defaultdict

from config import cfg
from model import make_model
from data.build_DG_dataloader import build_reid_test_loader
from utils.re_ranking import re_ranking


def normalize_class(c):
    c = str(c).strip().lower()
    mapping = {
        "trafficsignal": "trafficsign", "trafficsign": "trafficsign",
        "container": "container",
        "rubbishbins": "rubbishbin", "rubbishbin": "rubbishbin",
        "crosswalk": "crosswalk", "crosswalks": "crosswalk",
    }
    return mapping.get(c, c)


def load_class_map(csv_path):
    """Return {imageName: normalized_class}."""
    out = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        cls_idx = header.index("Class")
        name_idx = header.index("imageName")
        for row in reader:
            out[row[name_idx]] = normalize_class(row[cls_idx])
    return out


def compute_metrics(distmat, q_pids, g_pids, max_rank=20):
    """Standard mAP + Rank-k. Returns per-query AP and CMC."""
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0
    for q_idx in range(num_q):
        orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1
    assert num_valid_q > 0
    all_cmc = np.asarray(all_cmc).astype(np.float32).mean(axis=0)
    mAP = np.mean(all_AP)
    return mAP, all_cmc, all_AP


def extract_feats(model, loader, num_query):
    feats = []
    pids = []
    camids = []
    names = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img = batch['images'].cuda()
            pid = batch['targets']
            cam = batch['camids']
            path = batch['img_paths']
            # TTA: original + horizontal flip
            out1 = model(img).float()
            out2 = model(torch.flip(img, dims=[3])).float()
            f = out1 + out2
            f = f / f.norm(p=2, dim=1, keepdim=True)
            feats.append(f.cpu())
            pids.extend(pid.tolist() if torch.is_tensor(pid) else list(pid))
            camids.extend(cam.tolist() if torch.is_tensor(cam) else list(cam))
            names.extend([os.path.basename(p) for p in path])
    feats = torch.cat(feats, 0).numpy()
    return feats[:num_query], feats[num_query:], \
           np.array(pids[:num_query]), np.array(pids[num_query:]), \
           names[:num_query], names[num_query:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--rerank", action="store_true", help="apply k-reciprocal rerank")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # Force eval dataset
    val_loader, num_query = build_reid_test_loader(cfg, 'UAM_Eval')

    model = make_model(cfg, cfg.MODEL.NAME, num_class=1, camera_num=None, view_num=None)
    model.load_param(cfg.TEST.WEIGHT)
    model.cuda()
    model.eval()

    qf, gf, q_pids, g_pids, q_names, g_names = extract_feats(model, val_loader, num_query)

    print(f"qf: {qf.shape}, gf: {gf.shape}")

    # Distance
    q_g_dist = 1 - qf @ gf.T   # cosine distance (feats are L2-normed)
    if args.rerank:
        q_q_dist = 1 - qf @ qf.T
        g_g_dist = 1 - gf @ gf.T
        distmat = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3)
    else:
        distmat = q_g_dist

    # Overall metrics
    mAP, cmc, all_AP = compute_metrics(distmat, q_pids, g_pids)
    print(f"\n{'=' * 50}")
    print(f"OVERALL  mAP: {mAP:.4f}  R1: {cmc[0]:.4f}  R5: {cmc[4]:.4f}  R10: {cmc[9]:.4f}")
    print(f"{'=' * 50}")

    # Per-class breakdown using query_classes.csv
    q_class_csv = os.path.join(cfg.DATASETS.ROOT_DIR, 'UAMEXTERNAL', 'query_classes.csv')
    q_class_map = load_class_map(q_class_csv)
    q_classes = np.array([q_class_map.get(n, 'unknown') for n in q_names])

    print(f"\n{'Class':<15} {'N':>5} {'mAP':>8} {'R1':>8} {'R5':>8} {'R10':>8}")
    print('-' * 60)
    for cls in sorted(set(q_classes)):
        mask = q_classes == cls
        if mask.sum() == 0:
            continue
        cls_dist = distmat[mask]
        cls_qpids = q_pids[mask]
        cls_mAP, cls_cmc, _ = compute_metrics(cls_dist, cls_qpids, g_pids)
        print(f"{cls:<15} {mask.sum():>5} {cls_mAP:>8.4f} "
              f"{cls_cmc[0]:>8.4f} {cls_cmc[4]:>8.4f} {cls_cmc[9]:>8.4f}")