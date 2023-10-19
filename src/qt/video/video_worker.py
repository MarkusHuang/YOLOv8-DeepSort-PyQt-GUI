from PyQt5.QtCore import QThread, pyqtSignal
from src.models.detection.yolov8_detector_onnx import YoloDetector
from src.models.pose.yolov8_pose_onnx import PoseDetector
from src.models.segmentation.yolov8_seg_onnx import YOLOSeg
from src.models.tracking.deep_sort.deep_sort import DeepSort
from src.models.tracking.byte_track.byte_tracker import BYTETracker
from src.utils.general import ROOT, add_image_id
from src.utils.visualize import draw_results
import os
import cv2 as cv
import numpy as np
from PyQt5.QtGui import QImage


class FileProcessThread(QThread):
    send_thread_start_finish_flag = pyqtSignal(str)
    send_video_info = pyqtSignal(dict)
    send_ai_output = pyqtSignal(list)
    send_display_frame = pyqtSignal(QImage)
    send_play_progress = pyqtSignal(int)
    def __init__(self):
        super(FileProcessThread, self).__init__()
        self.thread_name = "FileProcessThread"
        self.threadFlag = False
    
    def set_start_config(self, video_path, ai_task, screen_size, model_name="yolov8n", tracker_name="deepsort", confidence_threshold=0.35, iou_threshold=0.45, frame_interval=0):
        self.threadFlag = True
        self.video_path = video_path
        self.ai_task = ai_task
        self.pause_process = False
        self.confi_thr = confidence_threshold
        self.iou_thr = iou_threshold
        self.model_name = model_name
        self.tracker_name = tracker_name
        self.frame_interval = frame_interval
        self.get_screen_size(screen_size)
        self._init_yolo()
        self._init_tracker()

    def set_iou_threshold(self, iou_threshold):
        self.iou_thr = iou_threshold
    
    def set_confidence_threshold(self, confidence_threshold):
        self.confi_thr = confidence_threshold
    
    def set_model_name(self, model_name):
        self.model_name = model_name
    
    def set_tracker_name(self, tracker_name):
        self.tracker_name = tracker_name
    
    def set_frame_interval(self, frame_interval):
        self.frame_interval = frame_interval
    
    def get_screen_size(self, screen_size):
        self.iw, self.ih = screen_size

    def _init_yolo(self):
        if self.ai_task == "object_detection":
            self.detector = YoloDetector()
            self.detector.init(
                model_path=os.path.join(ROOT, f"weights/detection/{self.model_name}.onnx"),
                class_txt_path=os.path.join(ROOT, "weights/classes.txt"),
                confidence_threshold=self.confi_thr,
                iou_threshold=self.iou_thr)
        elif self.ai_task == "pose_detection":
            self.pose_detector = PoseDetector()
            self.pose_detector.init(
                model_path=os.path.join(ROOT, f"weights/pose/{self.model_name}-pose.onnx"),
                confidence_threshold=self.confi_thr,
                iou_threshold=self.iou_thr)
        elif self.ai_task == "segmentation":
            self.seg_detector = YOLOSeg()
            self.seg_detector.init(
                model_path=os.path.join(ROOT, f"weights/segmentation/{self.model_name}-seg.onnx"),
                class_txt_path=os.path.join(ROOT, "weights/classes.txt"),
                confidence_threshold=self.confi_thr,
                iou_threshold=self.iou_thr)

    def _init_tracker(self):
        if self.tracker_name == "deepsort":
            self.tracker = DeepSort(
                model_path=os.path.join(ROOT, f"src/models/tracking/deep_sort/deep/checkpoint/ckpt.t7"))
        elif self.tracker_name == "bytetrack":
            self.tracker = BYTETracker(
                track_high_thresh=0.5, 
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                match_thresh=0.8,
                track_buffer=30,
                frame_rate=30)
    
    def stop_process(self):
        self.threadFlag = False
    
    def toggle_play_pause(self):
        self.pause_process = not self.pause_process
        
    def run(self):
        self.send_thread_start_finish_flag.emit("processing_on_file")
        media_fmt = self.check_image_or_video(self.video_path)
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_info = self.get_video_info(cap)
        self.send_video_info.emit(video_info)

        model_output = []
        frame_id = 1
        while self.threadFlag:
            if self.pause_process:
                continue
            ret, frame = cap.read()
            if ret is False:
                break

            if frame_id % int(self.frame_interval+1) == 0:
                model_output = []
                if self.ai_task == "object_detection":
                    model_output = self.detector.inference(frame, self.confi_thr, self.iou_thr)
                elif self.ai_task == "pose_detection":
                    model_output = self.pose_detector.inference(frame, self.confi_thr, self.iou_thr)
                elif self.ai_task == "segmentation":
                    model_output = self.seg_detector.inference(frame, self.confi_thr, self.iou_thr)
                
                if media_fmt == "video":
                    model_output = self.tracker.update(
                        detection_results=model_output,
                        ori_img=frame)
            model_output = add_image_id(model_output, frame_id)
            frame = draw_results(frame, model_output)
            display_frame = self.convert_cv_qt(frame, self.ih, self.iw)

            self.send_display_frame.emit(display_frame)
            self.send_play_progress.emit(int(frame_id/video_info["length"]*1000))
            self.send_ai_output.emit(model_output)
            frame_id += 1
        cap.release()
        if media_fmt == "video":
            blank_image = np.zeros((self.ih, self.iw, 3))
            blank_image = cv.cvtColor(blank_image.astype('uint8'), cv.COLOR_BGR2RGBA)
            show_image = QImage(blank_image.data, blank_image.shape[1], blank_image.shape[0], QImage.Format_RGBA8888)
            self.send_display_frame.emit(show_image)
            self.send_ai_output.emit([])
        self.send_thread_start_finish_flag.emit("waiting_for_setting")
    
    def get_video_info(self, video_cap):
        video_info = {}
        video_info["FPS"] = video_cap.get(cv.CAP_PROP_FPS)
        video_info["length"] = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
        video_info["size"] = (int(video_cap.get(cv.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        return video_info

    def check_image_or_video(self, media_path):
        img_fm = (".tif", ".tiff", ".jpg", ".jpeg", ".gif", ".png", ".eps", ".raw", ".cr2", ".nef", ".orf", ".sr2", ".bmp", ".ppm", ".heif")
        vid_fm = (".flv", ".avi", ".mp4", ".3gp", ".mov", ".webm", ".ogg", ".qt", ".avchd")
        media_fms = {"image": img_fm, "video": vid_fm}
        if any(media_path.lower().endswith(media_fms["image"]) for ext in media_fms["image"]):
            return "image"
        elif any(media_path.lower().endswith(media_fms["video"]) for ext in media_fms["video"]):
            return "video"
        else:
            raise TypeError("Please select an image or video")
    
    def convert_cv_qt(self, image, screen_height, screen_width):
        h, w, _ = image.shape
        scale = min(screen_width / w, screen_height / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv.resize(image, (nw, nh))
        image_paded = np.full(shape=[screen_height, screen_width, 3], fill_value=0)
        dw, dh = (screen_width - nw) // 2, (screen_height - nh) // 2
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
        image_paded = cv.cvtColor(image_paded.astype('uint8'), cv.COLOR_BGR2RGBA)
        return QImage(image_paded.data, image_paded.shape[1], image_paded.shape[0], QImage.Format_RGBA8888)