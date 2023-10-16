from PyQt5.QtCore import QThread, pyqtSignal
from src.models.detection.yolov8_detector_onnx import YoloDetector
from src.models.pose.yolov8_pose_onnx import PoseDetector
from src.models.segmentation.yolov8_seg_onnx import YOLOSeg
from src.models.tracking.deep_sort.deep_sort import DeepSort
from src.models.tracking.byte_track.byte_tracker import BYTETracker
from src.data_type.video_buffer import LatestFrame
from src.utils.general import ROOT, add_image_id
import os


class AiWorkerThread(QThread):
    send_ai_output = pyqtSignal(list)
    def __init__(self):
        super(AiWorkerThread, self).__init__()
        self.thread_name = "AiWorkerThread"
        self.threadFlag = False
    
    def set_start_config(self, ai_task, model_name="yolov8n", tracker_name="deepsort", confidence_threshold=0.35, iou_threshold=0.45):
        self.threadFlag = True
        self.ai_task = ai_task
        self.latest_frame = LatestFrame()
        self.confi_thr = confidence_threshold
        self.iou_thr = iou_threshold
        self.model_name = model_name
        self.tracker_name = tracker_name
        self._init_yolo()
        self._init_tracker()

    def set_iou_threshold(self, iou_threshold):
        self.iou_thr = iou_threshold
    
    def set_confidence_threshold(self, confidence_threshold):
        self.confi_thr = confidence_threshold
    
    def set_model_name(self, model_name):
        self.model_name = model_name

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
    
    def get_frame(self, frame_list):
        self.latest_frame.put(frame=frame_list[1],frame_id=frame_list[0],realtime=True)
    
    def stop_process(self):
        self.threadFlag = False
        
    def run(self):
        while self.threadFlag:
            frame_id, frame = self.latest_frame.get()
            if frame_id is None:
                break
            model_output = []
            if self.ai_task == "object_detection":
                model_output = self.detector.inference(frame, self.confi_thr, self.iou_thr)
            elif self.ai_task == "pose_detection":
                model_output = self.pose_detector.inference(frame, self.confi_thr, self.iou_thr)
            elif self.ai_task == "segmentation":
                model_output = self.seg_detector.inference(frame, self.confi_thr, self.iou_thr)

            model_output = self.tracker.update(
                detection_results=model_output,
                ori_img=frame)
            
            self.model_output = add_image_id(model_output, frame_id)
            self.send_ai_output.emit(model_output)
