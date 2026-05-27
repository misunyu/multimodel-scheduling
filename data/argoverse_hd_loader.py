"""Argoverse-HD streaming dataset loader (scaffold).

Argoverse-HD is the standard streaming-perception benchmark used with sap-toolkit.
Layout this loader assumes (download separately, place under data/argoverse_hd/):

    data/argoverse_hd/
        annotations/
            train.json, val.json
        rgb/
            <log_id>/
                ring_front_center/
                    <frame_idx>.jpg

The `StreamingFrameIterator` yields (timestamp, frame_bgr, gt_or_none) tuples at a
configurable nominal FPS (default 30 = Argoverse-HD's source rate). Drop frames if
the consumer falls behind — this mirrors how sap-toolkit measures streaming AP.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parent / "argoverse_hd"


@dataclass
class StreamSequence:
    log_id: str
    frame_paths: List[Path]
    annotations: Optional[dict] = None  # COCO-style image_id -> [ann]

    @property
    def n_frames(self) -> int:
        return len(self.frame_paths)


def discover_sequences(root: Path = DEFAULT_ROOT,
                       split: str = "val",
                       camera: str = "ring_front_center") -> List[StreamSequence]:
    """Scan the dataset root and return one StreamSequence per log."""
    rgb_root = root / "rgb"
    if not rgb_root.exists():
        raise FileNotFoundError(
            f"Argoverse-HD rgb root not found at {rgb_root}. "
            "Download from https://www.cs.cmu.edu/~mengtial/proj/streaming/ "
            "and extract under data/argoverse_hd/."
        )

    ann_path = root / "annotations" / f"{split}.json"
    coco = None
    if ann_path.exists():
        with ann_path.open() as f:
            coco = json.load(f)

    seqs: List[StreamSequence] = []
    for log_dir in sorted(rgb_root.iterdir()):
        if not log_dir.is_dir():
            continue
        cam_dir = log_dir / camera
        if not cam_dir.exists():
            continue
        frames = sorted(cam_dir.glob("*.jpg"))
        if not frames:
            continue
        seqs.append(StreamSequence(log_id=log_dir.name, frame_paths=frames, annotations=coco))
    return seqs


class StreamingFrameIterator:
    """Yields frames at a wall-clock paced rate, dropping late frames.

    Consumer pulls from `__iter__` — if the consumer is slower than `fps`, intermediate
    frames are dropped (the iterator catches up). This is the realistic mode used by
    sap-toolkit: the simulated camera does not block on the model.
    """

    def __init__(self, sequence: StreamSequence, fps: float = 30.0, real_time: bool = True):
        self.sequence = sequence
        self.fps = fps
        self.real_time = real_time
        self._period = 1.0 / fps if fps > 0 else 0.0

    def __iter__(self) -> Iterator[Tuple[float, np.ndarray, Optional[int]]]:
        t0 = time.time()
        for idx, path in enumerate(self.sequence.frame_paths):
            target_t = t0 + idx * self._period
            if self.real_time:
                now = time.time()
                if now < target_t:
                    time.sleep(target_t - now)
                elif now - target_t > self._period:
                    # Consumer is behind. Skip frames to catch up.
                    skip = int((now - target_t) / self._period)
                    if skip > 0 and idx + skip < self.sequence.n_frames:
                        path = self.sequence.frame_paths[idx + skip]

            frame = cv2.imread(str(path))
            if frame is None:
                continue
            yield (time.time() - t0, frame, idx)


def multi_stream_iterator(sequences: List[StreamSequence],
                          fps: float = 30.0) -> List[StreamingFrameIterator]:
    """Build N parallel StreamingFrameIterators — one per simulated camera/car.

    Drive these from separate threads/processes to simulate the multi-stream load
    that the scene-adaptive placement experiment requires.
    """
    return [StreamingFrameIterator(seq, fps=fps) for seq in sequences]
