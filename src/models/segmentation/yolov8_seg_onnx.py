import math
import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession
from src.models.segmentation.segmentation_base import SegmentBase, Model
from src.models.base.yolov8_base import ModelError
from src.utils.boxes import xywh2xyxy, nms
from src.utils.general import get_classes, sigmoid


class YOLOSeg(SegmentBase):
    def __init__(self):
        self._model = None

    def init(self, model_path, class_txt_path, confidence_threshold=0.5, iou_threshold=0.5):
        _class_names = get_classes(class_txt_path)
        _session = InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_names,self.output_names, input_size = self.get_onnx_model_details(_session)
        self.num_masks = 32
        self._model = Model(
            model=_session,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            input_size=input_size,
            class_names=_class_names)

        init_frame = np.random.randint(0, 256, (input_size[0], input_size[1], 3)).astype(np.uint8)
        self.inference(init_frame)

    def inference(self, image, confi_thres=None, iou_thres=None):
        if self._model is None:
            raise ModelError("Model not initialized. Have you called init()?")
        if confi_thres is None:
            confi_thres = self._model.confidence_threshold
        if iou_thres is None:
            iou_thres = self._model.iou_threshold
        image_size = (image.shape[1],image.shape[0])
        scale, processed_image = self.preprocess(image, self._model.input_size)
        outputs = self._model.model.run(self.output_names, {self.input_names[0]: processed_image})
        boxes, scores, class_ids, mask_pred = self.process_box_output(
            box_output=outputs[0], 
            scale=scale,
            image_size=image_size,
            confidence_threshold=confi_thres,
            iou_threshold=iou_thres)
        mask_maps = self.process_mask_output(mask_pred, boxes, outputs[1], image_size, scale)
        resutls = []
        for i in range(len(class_ids)):
            obj_dict = {
                "id": int(i),
                "class": self._model.class_names[int(class_ids[i])],
                "bbox": np.rint(boxes[i]),
                "confidence": scores[i],
                "keypoints":np.array([]),
                "segmentation": np.array(mask_maps[i])}
            resutls.append(obj_dict)
        return resutls
    
    def process_box_output(self, box_output, scale, image_size, confidence_threshold, iou_threshold):
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > confidence_threshold, :]
        scores = scores[scores > confidence_threshold]
        if len(scores) == 0:
            return [], [], [], np.array([])
        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)
        boxes = box_predictions[:, :4]
        boxes = xywh2xyxy(boxes)
        boxes *= scale
        boxes[:, 0] = np.clip(boxes[:, 0], 0, image_size[0])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, image_size[1])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, image_size[0])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, image_size[1])
        indices = nms(boxes, scores, iou_threshold)
        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def process_mask_output(self, mask_predictions, boxes, mask_output, image_size, scale):
        if mask_predictions.shape[0] == 0:
            return []
        mask_output = np.squeeze(mask_output)
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))
        scale_new = min((mask_height/image_size[1],mask_width/image_size[0]))
        scale_boxes = boxes * scale_new
        mask_maps = np.zeros((len(scale_boxes), image_size[1], image_size[0]))
        blur_size = (int(image_size[0] / mask_width), int(image_size[1] / mask_height))
        for i in range(len(scale_boxes)):
            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))
            x1 = int(math.floor(boxes[i][0]))
            y1 = int(math.floor(boxes[i][1]))
            x2 = int(math.ceil(boxes[i][2]))
            y2 = int(math.ceil(boxes[i][3]))
            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv.resize(
                scale_crop_mask,
                (x2 - x1, y2 - y1),
                interpolation=cv.INTER_CUBIC)
            crop_mask = cv.blur(crop_mask, blur_size)
            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask
        return mask_maps
