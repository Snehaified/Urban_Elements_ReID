# encoding: utf-8
"""
Combined UAMEXTERNAL + Urban2026 dataset for training.
Query/gallery point at UAM query/test with ground-truth objectIDs,
so train.py's per-epoch eval gives real per-class mAP on UAM c004.
"""

import csv
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class UAM_Urban_Combined(ImageDataset):
    """
    Expected layout (ROOT_DIR should be project root, i.e. contains both folders):
      <root>/UAMEXTERNAL/{image_train,image_query,image_test,train.csv,query.csv,test.csv}
      <root>/Urban2026/{image_train,train.csv}  (train_classes.csv used for IDs)
    """

    # PID namespace offsets to avoid collisions
    UAM_OFFSET = 0
    URBAN_OFFSET = 100000  # guaranteed > max UAM ID

    def __init__(self, root='.', verbose=True, **kwargs):
        self.root = root
        self.uam_dir = osp.join(root, 'UAMEXTERNAL')
        self.urban_dir = osp.join(root, 'Urban2026')

        self._check()

        # Training data: UAM train + Urban2026 train (combined)
        train_uam = self._read_train(
            csv_path=osp.join(self.uam_dir, 'train.csv'),
            img_dir=osp.join(self.uam_dir, 'image_train'),
            pid_offset=self.UAM_OFFSET,
        )
        train_urban = self._read_train(
            csv_path=osp.join(self.urban_dir, 'train.csv'),
            img_dir=osp.join(self.urban_dir, 'image_train'),
            pid_offset=self.URBAN_OFFSET,
        )
        train_raw = train_uam + train_urban

        # Global relabel across both sources
        pid_container = sorted({pid for _, pid, _ in train_raw})
        pid2label = {pid: idx for idx, pid in enumerate(pid_container)}
        train = [(p, pid2label[pid], cam) for p, pid, cam in train_raw]

        # Query/gallery from UAM (with ground-truth IDs, not relabeled)
        query = self._read_eval(
            csv_path=osp.join(self.uam_dir, 'query.csv'),
            img_dir=osp.join(self.uam_dir, 'image_query'),
        )
        gallery = self._read_eval(
            csv_path=osp.join(self.uam_dir, 'test.csv'),
            img_dir=osp.join(self.uam_dir, 'image_test'),
        )

        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print(f"[UAM_Urban_Combined] train: {len(train)} imgs, "
                  f"{len(pid_container)} pids (UAM+Urban combined)")
            print(f"[UAM_Urban_Combined] query (UAM c004): {len(query)} imgs")
            print(f"[UAM_Urban_Combined] gallery (UAM c001/c002/c003): {len(gallery)} imgs")

        super().__init__(self.train, self.query, self.gallery, **kwargs)

    def _check(self):
        for d in [self.uam_dir, self.urban_dir]:
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")
        for f in [
            osp.join(self.uam_dir, 'train.csv'),
            osp.join(self.uam_dir, 'query.csv'),
            osp.join(self.uam_dir, 'test.csv'),
            osp.join(self.urban_dir, 'train.csv'),
        ]:
            if not osp.exists(f):
                raise RuntimeError(f"CSV not found: {f}")

    def _read_train(self, csv_path, img_dir, pid_offset):
        """Returns list of (full_img_path, pid, camid). pid is source-specific int offset."""
        out = []
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                cam_str, img_name = row[0], row[1]
                pid_raw = row[2]  # UAM: objectID; Urban2026: Corresponding Indexes
                try:
                    pid = int(pid_raw) + pid_offset
                except (ValueError, TypeError):
                    continue
                camid = int(cam_str[1:])  # 'c001' -> 1
                out.append((osp.join(img_dir, img_name), pid, camid))
        return out

    def _read_eval(self, csv_path, img_dir):
        """Returns list of (full_img_path, pid, camid). Uses real UAM objectIDs (not relabeled)."""
        out = []
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            has_pid = len(header) >= 3
            for row in reader:
                cam_str, img_name = row[0], row[1]
                pid = int(row[2]) if has_pid else -1
                camid = int(cam_str[1:])
                out.append((osp.join(img_dir, img_name), pid, camid))
        return out