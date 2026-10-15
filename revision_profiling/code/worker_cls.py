"""Image classification worker.

Measured path (worker thread): resize (shorter side 256) -> centre crop 224 -> BGR->RGB -> /255 -> ImageNet mean/std
-> NCHW float32 tensor -> inference -> argmax. Decode is done beforehand (executor.load_inputs).
The two Caffe2-exported models (squeezenet1.0-12, vgg19) take BGR 0..255 with the Caffe channel means instead.
The original runtime used letterbox + /255 without mean/std (image_processing.image_preprocess); with that input the
classifiers do not predict the right class, so the standard ImageNet transform is used here (CHANGES.md).
"""
import sys

sys.dont_write_bytecode = True

import cv2
import numpy as np

from worker_base import Worker

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


CAFFE_MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32)


def cls_preprocess(img_bgr, size=224, resize=256, caffe=False):
    h, w = img_bgr.shape[:2]
    s = resize / min(h, w)
    nw, nh = max(size, int(round(w * s))), max(size, int(round(h * s)))
    img = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    img = img[y0:y0 + size, x0:x0 + size]
    if caffe:
        img = img.astype(np.float32) - CAFFE_MEAN_BGR
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None, ...])


class ClsWorker(Worker):
    def setup(self):
        i = self.session.get_inputs()[0]
        self.input_name = i.name
        self.output_name = self.session.get_outputs()[0].name
        self.w, self.h = self.spec["input_size"]
        assert self.spec["preprocess"] in ("cls_resize256_crop224_imagenet_norm", "cls_resize256_crop224_caffe_bgr_mean"), self.spec["preprocess"]
        self.caffe = self.spec["preprocess"].endswith("caffe_bgr_mean")
        assert self.spec["layout"] == "NCHW" and list(i.shape[1:]) == [3, self.h, self.w] and self.w == self.h, (i.shape, self.spec)

    def preprocess(self, idx):
        return cls_preprocess(self.inputs[idx % len(self.inputs)], self.w, caffe=self.caffe)

    def infer(self, x):
        return self.session.run([self.output_name], {self.input_name: x})[0]

    def postprocess(self, y, idx):
        return int(np.argmax(np.squeeze(y)))
