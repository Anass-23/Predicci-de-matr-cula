import time

class Timer:
    def __init__(self):
        self.start_time: float = None
        self.end_time: float   = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def reset(self):
        self.start_time = None
        self.end_time = None

    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0
        elif self.end_time is None:
            return time.time() - self.start_time
        else:
            return self.end_time - self.start_time