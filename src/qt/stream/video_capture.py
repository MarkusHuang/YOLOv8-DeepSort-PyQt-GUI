import os
from PyQt5.QtCore import QThread, pyqtSignal
import cv2 as cv
import datetime


class CameraCaptureThread(QThread):
    send_video_info = pyqtSignal(dict)
    send_frame = pyqtSignal(list)
    def __init__(self):
        super(CameraCaptureThread, self).__init__()
        self.thread_name = "CameraCaptureThread"
        self.threadFlag = False
    
    def set_start_config(self, video_source):
        self.threadFlag = True
        self.get_video_source(video_source)
    
    def get_video_source(self, video_source):
        self.video_source = video_source
    
    def get_video_info(self, video_cap):
        video_info = {}
        video_info["FPS"] = video_cap.get(cv.CAP_PROP_FPS)
        video_info["length"] = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
        video_info["size"] = (int(video_cap.get(cv.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        return video_info
    
    def stop_capture(self):
        self.threadFlag = False

    def run(self):        
        cap = cv.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_info = self.get_video_info(cap)
        self.send_video_info.emit(video_info)

        idx_frame = 0
        while self.threadFlag:
            ret, frame = cap.read()
            if ret is False or self.threadFlag is False:
                break
            self.send_frame.emit(list([idx_frame,frame]))
            idx_frame += 1
        self.send_frame.emit(list([None,None]))
        cap.release()




















########################

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class VideoCap2(QThread):
    def __init__(self):
        super(VideoCap2, self).__init__()
        self.thread_name = "VideoCapThread"
        self.threadFlag = False
    
    def set_start_config(self, video_source, frame_buffer, current_frame, save_dir="OUTPUT", resize=None, preprocess=None):
        self.video_source = video_source
        self.save_dir = save_dir
        self.frame_buffer = frame_buffer
        self.current_frame = current_frame
        self.resize = resize
        self._preprocess = preprocess
        self.threadFlag = True
    
    def _resize(self, img, resize_size):
        h,w = img.shape[0], img.shape[1]
        img_resize = cv.resize(img, (resize_size[0], int(resize_size[1]*h/w)), interpolation=cv.INTER_NEAREST)
        return img_resize
    
    def _preprocess(self, img):
        processed_img = img
        return processed_img

    def run(self):
        if isinstance(self.video_source, int) or "rtsp" in self.video_source:
            mp4_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.mp4')
            mp4_path = os.path.join(self.save_dir, mp4_name)
            mkdir(f"{self.save_dir}")
        
        cap = cv.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.stop_flag = True
            raise IOError("Couldn't open webcam or video")
        else:
            self.stop_flag = False
        if self.resize is not None:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, self.resize[1])
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.resize[0])
        
        video_FourCC = cv.VideoWriter_fourcc(*'mp4v')
        video_fps = cap.get(cv.CAP_PROP_FPS)
        
        video_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        video_writer = cv.VideoWriter(mp4_path, video_FourCC, video_fps, video_size)

        idx_frame = 0
        while True:
            ret, frame = cap.read()
            if ret is False or self.stop_flag is True:
                break
            video_writer.write(frame)

            self.frame_buffer.put(frame, idx_frame, True)

            if self._preprocess:
                frame = self._preprocess(frame)
            self.current_frame.put(frame, idx_frame, True)
                      
            idx_frame += 1

            if idx_frame >= 400:
                self.stop_flag = True
        cap.release()
        video_writer.release()

