# encoding: utf-8
"""UAMEXTERNAL query+test evaluation, with ground-truth IDs."""

import csv
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class UAM_Eval(ImageDataset):
    def __init__(self, root='.', verbose=True, **kwargs):
        self.uam_dir = osp.join(root, 'UAMEXTERNAL')
        if not osp.exists(self.uam_dir):
            raise RuntimeError(f"'{self.uam_dir}' is not available")

        # Minimal train (needed by base class) — use UAM train with real IDs + relabel
        train_raw = self._read(
            osp.join(self.uam_dir, 'train.csv'),
            osp.join(self.uam_dir, 'image_train'),
        )
        pid_container = sorted({pid for _, pid, _ in train_raw})
        pid2label = {pid: idx for idx, pid in enumerate(pid_container)}
        train = [(p, pid2label[pid], cam) for p, pid, cam in train_raw]

        query = self._read(
            osp.join(self.uam_dir, 'query.csv'),
            osp.join(self.uam_dir, 'image_query'),
        )
        gallery = self._read(
            osp.join(self.uam_dir, 'test.csv'),
            osp.join(self.uam_dir, 'image_test'),
        )

        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print(f"[UAM_Eval] query: {len(query)} imgs, gallery: {len(gallery)} imgs")

        super().__init__(self.train, self.query, self.gallery, **kwargs)

    def _read(self, csv_path, img_dir):
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