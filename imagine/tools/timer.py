"""
Timer class is designed for time recording
"""
import time
from imagine.tools.icy_decorator import icy


@icy
class Timer(object):
    """
    Class designed for time recording

    Simply provide an event name to the `tick` method to start recording.
    The `tock` method stops the recording and the `record` property allow
    one to access the recorded time.
    """
    def __init__(self):
        self._record = dict()

    @property
    def record(self):
        """
        Dictionary of recorded times using event name as keys
        """
        return self._record

    @record.setter
    def record(self, record):
        raise NotImplementedError

    def tick(self, event):
        """
        Starts timing of a given event

        Parameters
        ----------
        event : str
            event name (will be key of the record attribute)
        """
        assert isinstance(event, str)
        self._record[event] = time.perf_counter()

    def tock(self, event):
        """
        Stops timing

        Parameters
        ----------
        event : str
            event name (will be key of the record attribute)
        """
        assert isinstance(event, str)
        assert (event in self._record.keys())
        self._record[event] = time.perf_counter() - self._record[event]
        return self._record[event]
