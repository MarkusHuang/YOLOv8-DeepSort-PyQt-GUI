from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from src.data_type.video_buffer import FrameBuffer
import cv2 as cv
import numpy as np
from src.utils.visualize import draw_results
import copy


class VideoVisualizationThread(QThread):
    send_thread_start_stop_flag = pyqtSignal(str)
    send_displayable_frame = pyqtSignal(QImage)
    send_ai_output = pyqtSignal(list)
    def __init__(self):
        super(VideoVisualizationThread, self).__init__()
        self.thread_name = "VideoVisualizationThread"
        self.threadFlag = False
    
    def set_start_config(self, screen_size):
        self.threadFlag = True
        self.frame_buffer = FrameBuffer(10)
        self.ai_output = []
        self.get_screen_size(screen_size)
    
    def get_fresh_frame(self, frame_list):
        self.frame_buffer.put(frame=copy.deepcopy(frame_list[1]), frame_id=frame_list[0], realtime=True)

    def get_ai_output(self, ai_output):
        self.ai_output = copy.deepcopy(ai_output)
    
    def get_screen_size(self, screen_size):
        self.iw, self.ih = screen_size
    
    def stop_display(self):
        self.threadFlag = False

    def run(self):
        self.send_thread_start_stop_flag.emit("processing_on_camera")
        while self.threadFlag:
            frame_id, frame = self.frame_buffer.get()
            if frame_id is not None:
                frame = draw_results(frame, self.ai_output)
                show_image = self.convert_cv_qt(frame, self.ih, self.iw)
                self.send_displayable_frame.emit(show_image)
                self.send_ai_output.emit(self.ai_output)
            else:
                break
        blank_image = np.zeros((self.ih, self.iw, 3))
        blank_image = cv.cvtColor(blank_image.astype('uint8'), cv.COLOR_BGR2RGBA)
        show_image = QImage(blank_image.data, blank_image.shape[1], blank_image.shape[0], QImage.Format_RGBA8888)
        self.send_displayable_frame.emit(show_image)
        self.send_ai_output.emit([])
        self.send_thread_start_stop_flag.emit("waiting_for_setting")


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
    
    
