import numpy as np
import os
import cv2 as cv
from collections import namedtuple
import abc
from src.utils.general import path_leaf
import time

class ModelError(Exception):
    pass

Model = namedtuple("Model", "model input_size")

class YoloPredictorBase(object):
    def __init__(self):
        self._model = None
    
    @abc.abstractmethod
    def init(self):
        return NotImplemented
    
    @abc.abstractmethod
    def postprocess(self, model_output, scale):
        return NotImplemented

    @abc.abstractmethod
    def inference(self, image):
        return NotImplemented
    
    @abc.abstractstaticmethod
    def draw_results(self, image, model_results):
        return NotImplemented
    
    @staticmethod
    def preprocess(original_image, input_size):
        [height, width, _] = original_image.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image
        scale = length / input_size[0]
        processed_image = cv.dnn.blobFromImage(image, scalefactor=1 / 255, size=(input_size[0], input_size[1]))
        return scale, processed_image

    def inference_image(self, image_path, save_dir=None, display=False):
        image = cv.imread(image_path)
        model_results = self.inference(image)
        image = self.draw_results(image, model_results)
        if save_dir:
            image_name = path_leaf(image_path).replace(".", "_processed.")
            image_save_path = os.path.join(save_dir, image_name)
            cv.imwrite(image_save_path, image)
        if display:
            cv.namedWindow("YOLOv8", cv.WINDOW_NORMAL)
            cv.imshow("YOLOv8", image)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def inference_video(self, video_path, save_dir=None, display=False):
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if save_dir:
            video_name = path_leaf(video_path).replace(".", "_processed.")
            video_save_path = os.path.join(save_dir, video_name)
            video_FourCC = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_fps = cap.get(cv.CAP_PROP_FPS)
            video_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
            video_writer = cv.VideoWriter(video_save_path, video_FourCC, video_fps, video_size)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            start = time.time()
            results = self.inference(frame)
            end = time.time()
            frame = self.draw_results(frame, results)            
            cv.putText(frame, "FPS= "+str(int(1 / (end - start))), (0, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            print("[{:.01f}%] time: {:.03f}s, FPS: {:.01f}".format((frame_id+1)/video_length*100, end - start, 1 / (end - start)), end="\r")
            if save_dir:
                video_writer.write(frame)
            if display:
                cv.namedWindow("YOLOv8", cv.WINDOW_NORMAL)
                cv.imshow("YOLOv8", frame)
                if cv.waitKey(1) & 0xFF == ord('\x1b'):
                    break
            frame_id += 1
        cap.release()
        cv.destroyAllWindows()
    
    @staticmethod
    def get_onnx_model_details(onnx_session):
        model_inputs = onnx_session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        input_shape = model_inputs[0].shape
        input_height = input_shape[2]
        input_width = input_shape[3]
        input_size = (input_width,input_height)
        model_outputs = onnx_session.get_outputs()
        output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        return input_names, output_names, input_size
