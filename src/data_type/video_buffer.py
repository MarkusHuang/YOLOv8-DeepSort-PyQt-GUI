import queue

class LatestFrame:
    def __init__(self):
        self.frame = queue.Queue(maxsize=1)
        self.frame_id = queue.Queue(maxsize=1)
    
    def clear_buffer(self):
        with self.frame.mutex, self.frame_id.mutex:
            self.frame.queue.clear()
            self.frame_id.queue.clear()
    
    def put(self, frame, frame_id, realtime=False):
        if self.frame.full() and realtime is True:
            self.clear_buffer()
        self.frame.put(frame, block=True, timeout=None)
        self.frame_id.put(frame_id, block=True, timeout=None)
    
    def get(self): 
        frame_tmp = self.frame.get(block=True, timeout=None)
        id_tmp = self.frame_id.get(block=True, timeout=None)
        return id_tmp, frame_tmp


class FrameBuffer:
    def __init__(self, buffer_size:int) -> None:
        self.frame_buffer = queue.Queue(maxsize=buffer_size) 
        self.frame_ids = queue.Queue(maxsize=buffer_size)
    
    def put(self, frame, frame_id, realtime=False):
        if self.frame_buffer.full() and realtime:
            tmp=self.frame_buffer.get(block=False, timeout=None)
            tmp=self.frame_ids.get(block=False, timeout=None)
        self.frame_buffer.put(frame, block=True, timeout=None)
        self.frame_ids.put(frame_id, block=True, timeout=None)
        
    def get(self):
        try:
            frame_tmp = self.frame_buffer.get(block=True, timeout=4)
            id_tmp = self.frame_ids.get(block=True, timeout=4)
            return id_tmp, frame_tmp
        except:
            return None, None
        
    def clear_buffer(self):
        with self.frame_buffer.mutex, self.frame_ids.mutex:
            self.frame_buffer.queue.clear()
            self.frame_ids.queue.clear()