"""
recording time
"""

import time

from imagine.tools.icy_decorator import icy


@icy
class Timer(object):

    def __init__(self):
        self._record = dict()

    @property
    def record(self):
        return self._record

    @record.setter
    def record(self, record):
        raise NotImplementedError

    def tick(self, event):
        assert isinstance(event, str)
        self._record[event] = time.perf_counter()

    def tock(self, event):
        assert isinstance(event, str)
        assert (event in self._record.keys())
        self._record[event] = time.perf_counter() - self._record[event]
