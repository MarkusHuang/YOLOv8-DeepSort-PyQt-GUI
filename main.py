from src.qt.stream.video_capture import CameraCaptureThread
from src.qt.stream.visualize import VideoVisualizationThread
from src.qt.stream.ai_worker import AiWorkerThread
from src.ui.main_window import Ui_MainWindow
from src.qt.video.video_worker import FileProcessThread
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
import sys
import numpy as np


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.ai_thread = AiWorkerThread()
        self.camera_thread = CameraCaptureThread()
        self.display_thread = VideoVisualizationThread()
        self.file_process_thread = FileProcessThread()

        self.conf_thr = 0.3
        self.iou_thr = 0.45
        self.frame_interval = 0
        self.model_name = "yolov8n"
        self.ai_task = "object_detection"
        
        self.init_slots()
        self.buttons_states("waiting_for_setting")
    
    def init_slots(self):
        self.radioButton_det.toggled.connect(lambda: self.get_ai_task(self.radioButton_det))
        self.radioButton_pose.toggled.connect(lambda: self.get_ai_task(self.radioButton_pose))
        self.radioButton_seg.toggled.connect(lambda: self.get_ai_task(self.radioButton_seg))
        self.doubleSpinBox_conf.valueChanged.connect(lambda x: self.update_parameter(x, 'doubleSpinBox_conf'))
        self.doubleSpinBox_interval.valueChanged.connect(lambda x: self.update_parameter(x, 'doubleSpinBox_interval'))
        self.doubleSpinBox_iou.valueChanged.connect(lambda x: self.update_parameter(x, 'doubleSpinBox_iou'))
        self.horizontalSlider_conf.valueChanged.connect(lambda x: self.update_parameter(x, 'horizontalSlider_conf'))
        self.horizontalSlider_interval.valueChanged.connect(lambda x: self.update_parameter(x, 'horizontalSlider_interval'))
        self.horizontalSlider_iou.valueChanged.connect(lambda x: self.update_parameter(x, 'horizontalSlider_iou'))
        self.comboBox_model.currentTextChanged.connect(self.choose_model)
        self.pushButton_cam.clicked.connect(self.process_camera)
        self.pushButton_file.clicked.connect(self.process_file)
        self.pushButton_stop.clicked.connect(self.stop_video)
        self.pushButton_play.clicked.connect(self.file_process_thread.toggle_play_pause)

    
    def resizeEvent(self, event:QtGui.QResizeEvent):
        self.screen_size = (self.label_display.width(), self.label_display.height())
        self.display_thread.get_screen_size(self.screen_size)
        self.file_process_thread.get_screen_size(self.screen_size)
        QtWidgets.QMainWindow.resizeEvent(self, event)
    

    def update_parameter(self, x, flag):
        if flag == 'doubleSpinBox_conf':
            self.horizontalSlider_conf.setValue(int(x*100))
            self.conf_thr = float(x)
        elif flag == 'doubleSpinBox_interval':
            self.horizontalSlider_interval.setValue(int(x))
            self.frame_interval = int(x)
            self.file_process_thread.set_frame_interval(self.frame_interval)
        elif flag == 'doubleSpinBox_iou':
            self.horizontalSlider_iou.setValue(int(x*100))
            self.iou_thr = float(x)
        elif flag == 'horizontalSlider_conf':
            self.doubleSpinBox_conf.setValue(x/100)
            self.conf_thr = float(x/100)
        elif flag == 'horizontalSlider_interval':
            self.doubleSpinBox_interval.setValue(x)
            self.frame_interval = int(x)
            self.file_process_thread.set_frame_interval(self.frame_interval)
        elif flag == 'horizontalSlider_iou':
            self.doubleSpinBox_iou.setValue(x/100)
            self.iou_thr = float(x/100)
        if self.ai_thread.isRunning:
            self.ai_thread.set_confidence_threshold(self.conf_thr)
            self.ai_thread.set_iou_threshold(self.iou_thr)
        if self.file_process_thread.isRunning:
            self.file_process_thread.set_confidence_threshold(self.conf_thr)
            self.file_process_thread.set_iou_threshold(self.iou_thr)

    def get_ai_task(self, btn):
        if btn.text() == 'Detection':
            if btn.isChecked() == True:
                self.ai_task = "object_detection"
        elif btn.text() == 'Pose Estimation':
            if btn.isChecked() == True:
                self.ai_task = "pose_detection"
        elif btn.text() == 'Segmentation':
            if btn.isChecked() == True:
                self.ai_task = "segmentation"
    
    def choose_model(self):
        self.model_name = self.comboBox_model.currentText()
        self.model_name = self.model_name.lower()
    
    def buttons_states(self, work_state):
        if work_state == "waiting_for_setting":
            self.radioButton_det.setDisabled(False)
            self.radioButton_pose.setDisabled(False)
            self.radioButton_seg.setDisabled(False)
            self.comboBox_model.setDisabled(False)
            self.pushButton_cam.setDisabled(False)
            self.pushButton_file.setDisabled(False)
            self.pushButton_play.setDisabled(True)
            self.pushButton_stop.setDisabled(True)
            self.doubleSpinBox_conf.setDisabled(False)
            self.horizontalSlider_conf.setDisabled(False)
            self.doubleSpinBox_interval.setDisabled(False)
            self.horizontalSlider_interval.setDisabled(False)
            self.doubleSpinBox_iou.setDisabled(False)
            self.horizontalSlider_iou.setDisabled(False)
            self.doubleSpinBox_interval.setDisabled(False)
            self.horizontalSlider_interval.setDisabled(False)
        elif work_state == "processing_on_camera":
            self.pushButton_play.click
            self.radioButton_det.setDisabled(True)
            self.radioButton_pose.setDisabled(True)
            self.radioButton_seg.setDisabled(True)
            self.comboBox_model.setDisabled(True)
            self.pushButton_cam.setDisabled(True)
            self.pushButton_file.setDisabled(True)
            self.pushButton_play.setDisabled(True)
            self.pushButton_stop.setDisabled(False)
            self.doubleSpinBox_conf.setDisabled(False)
            self.horizontalSlider_conf.setDisabled(False)
            self.doubleSpinBox_interval.setDisabled(True)
            self.horizontalSlider_interval.setDisabled(False)
            self.doubleSpinBox_iou.setDisabled(False)
            self.horizontalSlider_iou.setDisabled(False)
            self.doubleSpinBox_interval.setDisabled(True)
            self.horizontalSlider_interval.setDisabled(True)
        elif work_state == "processing_on_file":
            self.radioButton_det.setDisabled(True)
            self.radioButton_pose.setDisabled(True)
            self.radioButton_seg.setDisabled(True)
            self.comboBox_model.setDisabled(True)
            self.pushButton_cam.setDisabled(True)
            self.pushButton_file.setDisabled(True)
            self.pushButton_play.setDisabled(False)
            self.pushButton_stop.setDisabled(False)
            self.doubleSpinBox_conf.setDisabled(False)
            self.horizontalSlider_conf.setDisabled(False)
            self.doubleSpinBox_interval.setDisabled(False)
            self.horizontalSlider_interval.setDisabled(False)
            self.doubleSpinBox_iou.setDisabled(False)
            self.horizontalSlider_iou.setDisabled(False)
            self.doubleSpinBox_interval.setDisabled(False)
            self.horizontalSlider_interval.setDisabled(False)
    
    def process_camera(self):
        self.ai_thread.set_start_config(
            ai_task=self.ai_task,
            model_name=self.model_name)
        
        self.camera_thread.set_start_config(video_source=0)
        self.display_thread.set_start_config([self.label_display.width(),self.label_display.height()])

        self.camera_thread.send_frame.connect(self.display_thread.get_fresh_frame)
        self.camera_thread.send_frame.connect(self.ai_thread.get_frame)
        self.ai_thread.send_ai_output.connect(self.display_thread.get_ai_output)
        self.display_thread.send_displayable_frame.connect(self.update_display_frame)
        self.display_thread.send_ai_output.connect(self.update_statistic_table)
        self.display_thread.send_thread_start_stop_flag.connect(self.buttons_states)

        self.ai_thread.start()
        self.display_thread.start()
        self.camera_thread.start()

        
    def process_file(self):
        img_fm = (".tif", ".tiff", ".jpg", ".jpeg", ".gif", ".png", ".eps", ".raw", ".cr2", ".nef", ".orf", ".sr2", ".bmp", ".ppm", ".heif")
        vid_fm = (".flv", ".avi", ".mp4", ".3gp", ".mov", ".webm", ".ogg", ".qt", ".avchd")
        file_list = " *".join(img_fm+vid_fm)
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "choose an image or video file", "./data", f"Files({file_list})")
        if file_name:
            self.file_process_thread.set_start_config(
                video_path=file_name,
                ai_task=self.ai_task,
                screen_size=[self.label_display.width(),self.label_display.height()],
                model_name=self.model_name,
                confidence_threshold=self.conf_thr,
                iou_threshold=self.iou_thr,
                frame_interval=self.frame_interval)
            self.file_process_thread.send_ai_output.connect(self.update_statistic_table)
            self.file_process_thread.send_display_frame.connect(self.update_display_frame)
            self.file_process_thread.send_play_progress.connect(self.progressBar_play.setValue)
            self.file_process_thread.send_thread_start_finish_flag.connect(self.buttons_states)
            self.file_process_thread.start()

    def stop_video(self):
        self.display_thread.stop_display()
        self.ai_thread.stop_process()
        self.camera_thread.stop_capture()
        self.file_process_thread.stop_process()

    def update_display_frame(self, showImage):
        self.label_display.setPixmap(QtGui.QPixmap.fromImage(showImage))
    
    def clean_table(self):
        while (self.tableWidget_results.rowCount() > 0):
            self.tableWidget_results.removeRow(0)

    def update_statistic_table(self, ai_output):
        self.clean_table()
        self.tableWidget_results.setRowCount(0)
        if ai_output == []:
            return
        for box in ai_output:
            each_item = [str(box["id"]),str(box["class"]), "{:.1f}%".format(box["confidence"]*100), str(box["bbox"])]
            row = self.tableWidget_results.rowCount()
            self.tableWidget_results.insertRow(row)
            for j in range(len(each_item)):
                item = QtWidgets.QTableWidgetItem(str(each_item[j]))
                item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.tableWidget_results.setItem(row, j, item)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())



