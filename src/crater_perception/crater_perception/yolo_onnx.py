"""YOLOv8 ONNX runner: letterbox -> onnxruntime (CPU) -> decode -> class-aware NMS.

Class names are read from the ONNX metadata written by the ultralytics exporter, so a
retrained model with other classes needs no code change.

Runtime: onnxruntime. Nodes that use this are launched with ~/crater_ws/.venv-ai/bin/python
(system ROS packages + onnxruntime, numpy pinned to the system's 1.26) because OpenCV 4.6's
DNN module cannot load YOLOv8 graphs.
"""
import ast

import cv2
import numpy as np


class Yolo:
    def __init__(self, path, imgsz, threads, fallback_names=('rock', 'meteor', 'crater', 'station')):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        meta = self.sess.get_modelmeta().custom_metadata_map
        try:
            names = ast.literal_eval(meta['names'])
            self.names = [names[i] for i in sorted(names)]
        except (KeyError, ValueError, SyntaxError):
            self.names = list(fallback_names)

    def __call__(self, bgr, conf_thres, iou_thres, min_box_px=0.0):
        """-> [(class_name, score, x0, y0, x1, y1)] in original image pixels, clipped
        (plain Python floats: ROS message fields reject numpy scalars)."""
        h0, w0 = bgr.shape[:2]
        s = self.imgsz / max(h0, w0)
        nh, nw = int(round(h0 * s)), int(round(w0 * s))
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)
        pred = self.sess.run(None, {self.inp: blob})[0][0].T          # (N, 4 + nc)
        scores = pred[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf > conf_thres                                       # low confidence -> dropped
        pred, cls, conf = pred[keep], cls[keep], conf[keep]
        if not len(pred):
            return []
        xywh = pred[:, :4].copy()
        xywh[:, 0] = (xywh[:, 0] - left) / s - xywh[:, 2] / s / 2
        xywh[:, 1] = (xywh[:, 1] - top) / s - xywh[:, 3] / s / 2
        xywh[:, 2:] /= s
        offs = xywh.copy()
        offs[:, :2] += cls[:, None] * 4096                             # class-aware NMS
        idx = cv2.dnn.NMSBoxes(offs.tolist(), conf.tolist(), conf_thres, iou_thres)
        out = []
        for i in np.array(idx).flatten():
            x, y, w, h = xywh[i]
            x0, y0 = max(0.0, x), max(0.0, y)                          # clip to the image
            x1, y1 = min(float(w0), x + w), min(float(h0), y + h)
            if x1 - x0 < min_box_px or y1 - y0 < min_box_px:
                continue
            out.append((self.names[int(cls[i])], float(conf[i]), float(x0), float(y0), float(x1), float(y1)))
        return out
