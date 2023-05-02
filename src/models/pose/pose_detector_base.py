from collections import namedtuple
import cv2 as cv
from src.models.base.yolov8_base import YoloPredictorBase
from src.utils.visualize import PALLETE, draw_keypoints
import math

Model = namedtuple("Model", "model confidence_threshold iou_threshold input_size")

class PoseDetectorBase(YoloPredictorBase):
    @staticmethod
    def draw_results(image, model_results):
        img_cpy = image.copy()
        if model_results == []:
            return img_cpy
        height, width, _ = img_cpy.shape
        for obj in model_results:
            x0 = obj["bbox"][0]
            y0 = obj["bbox"][1]
            x1 = obj["bbox"][2]
            y1 = obj["bbox"][3]
            
            id = int(obj["id"])
            #confi = float(obj["confidence"])
            color = PALLETE[id%PALLETE.shape[0]]
            text = '%d'%(id)
            txt_color_light = (255, 255, 255)
            txt_color_dark = (0, 0, 0)
            font = cv.FONT_HERSHEY_SIMPLEX
            FONT_SCALE = 1e-3 
            THICKNESS_SCALE = 6e-4 
            font_scale = min(width, height) * FONT_SCALE
            if font_scale <= 0.4:
                font_scale = 0.41 
            elif font_scale > 2:
                font_scale = 2.0
            thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
            txt_size = cv.getTextSize(text, font, 0.4, 1)[0]
            cv.rectangle(img_cpy, (x0, y0), (x1, y1), color, int(thickness*5*font_scale))
            cv.rectangle(
                img_cpy,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                color,
                -1)
            cv.putText(img_cpy, text, (x0, y0 + txt_size[1]), font, font_scale, txt_color_dark, thickness=thickness+1)
            cv.putText(img_cpy, text, (x0, y0 + txt_size[1]), font, font_scale, txt_color_light, thickness=thickness) 
            img_cpy = draw_keypoints(image, obj["keypoints"], color) 
        return img_cpy
