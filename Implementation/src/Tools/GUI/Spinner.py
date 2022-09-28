import sys
import threading
import time

class Spinner(threading.Thread):
    """A class for visualizing a CLI spinner that is shown during processing tasks. """

    def __init__(self):
        super().__init__(target = self._spin)
        super().setDaemon(True)
        self._stop_event = threading.Event()

    def stop(self):
        sys.stdout.write("\b")
        sys.stdout.write("\n")
        self._stop_event.set()

    def _spin(self):
        while not self._stop_event.isSet():
            for s in '|/-\\':
                sys.stdout.write(s)
                sys.stdout.flush()
                time.sleep(0.3)
                sys.stdout.write("\b")
