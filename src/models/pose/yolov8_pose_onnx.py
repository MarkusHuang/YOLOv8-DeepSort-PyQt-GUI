import numpy as np
from onnxruntime import InferenceSession
from src.utils.boxes import multiclass_nms_class_agnostic_keypoints
from src.models.pose.pose_detector_base import PoseDetectorBase, Model
from src.models.base.yolov8_base import ModelError


class PoseDetector(PoseDetectorBase):
    def __init__(self):
        self._model = None
    
    def init(self, model_path, confidence_threshold=0.3, iou_threshold=0.45):
        _session = InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_names,self.output_names, input_size = self.get_onnx_model_details(_session)
        self._model = Model(
            model=_session,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            input_size=input_size
        )

    def postprocess(self, model_output, scale, iou_thres, confi_thres):
        preds = np.squeeze(model_output[0]).T
        boxes = preds[:,:4]
        scores = preds[:,4:5]
        kpts = preds[:,5:]
        dets = multiclass_nms_class_agnostic_keypoints(boxes, scores, kpts, iou_thres, confi_thres)
        pose_results = []
        if dets is not None:
            for i, pred in enumerate(dets):
                bbox = pred[:4]#xywh2xyxy(pred[:4])
                bbox *= scale
                bbox = np.rint(bbox)
                kpts = pred[6:]
                kpt = (kpts.reshape((17,3)))*[scale,scale,1]
                pose_dict = {
                    "id":int(i),
                    "class":"person",
                    "confidence":pred[4],
                    "bbox":bbox,
                    "keypoints":np.array(kpt),
                    "segmentation":np.array([])}
                pose_results.append(pose_dict)
        return pose_results
    
    def inference(self, image, confi_thres=None, iou_thres=None):
        if self._model is None:
            raise ModelError("Model not initialized. Have you called init()?")
        if confi_thres is None:
            confi_thres = self._model.confidence_threshold
        if iou_thres is None:
            iou_thres = self._model.iou_threshold

        scale, meta = self.preprocess(image, self._model.input_size)
        model_input = {self.input_names[0]: meta}
        model_output = self._model.model.run(self.output_names, model_input)[0]
        pose_results = self.postprocess(model_output, scale, iou_thres, confi_thres)
        return pose_results
