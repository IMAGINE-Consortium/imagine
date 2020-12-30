"""
Timer class is designed for time recording.
"""

# %% IMPORTS
# Built-in imports
import time

# All declaration
__all__ = ['Timer']


# %% CLASS DEFINITIONS
class Timer(object):
    """
    Class designed for time recording.

    Simply provide an event name to the `tick` method to start recording.
    The `tock` method stops the recording and the `record` property allow
    one to access the recorded time.
    """
    def __init__(self):
        self._record = dict()

    @property
    def record(self):
        """
        Dictionary of recorded times using event name as keys.
        """
        return self._record

    @record.setter
    def record(self, record):
        raise NotImplementedError

    def tick(self, event):
        """
        Starts timing with a given event name.

        Parameters
        ----------
        event : str
            Event name (will be key of the record attribute).
        """
        self._record[event] = time.perf_counter()

    def tock(self, event):
        """
        Stops timing of the given event.

        Parameters
        ----------
        event : str
            Event name (will be key of the record attribute).
        """
        assert (event in self._record.keys())
        self._record[event] = time.perf_counter() - self._record[event]
        return self._record[event]
