import cv2
import logging
import numpy as np


module_logger = logging.getLogger("app.timers")


def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


class Timer:
    name = ""
    total = None

    _start = 0
    _auto_print = True

    def __init__(self, name, auto_print=True, log_level=logging.DEBUG):
        self.name = name
        self._auto_print = auto_print
        self.logger = logging.getLogger("app.timers.Timer")
        self.log_level = log_level

    def __enter__(self):
        self._start = clock()
        if self.total is None:
            self.total = 0
            if self._auto_print:
                self.logger.log(self.log_level, "Started timer on {}... ".format(self.name))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.total += clock() - self._start
        if self._auto_print:
            self.logger.log(self.log_level, "{} complete in {} ms ".format(self.name, self))

    def __str__(self):
        return "{:.2f}".format(self.total * 1000)


def iter_timer(seq, title=None, print_iterations=True, log_level=logging.DEBUG):
    """ Measures and prints total execution time, time of each iteration and some statistics"""
    logger = logging.getLogger("app.timers")

    def print_iter(idx, time):
        logger.log(log_level, " iter {:3}, time: {:.3} s".format(idx, time))

    if title is not None:
        logger.log(log_level, "Started timer on iterative {}".format(title))

    prev_time = None
    times = []
    idx = -1
    for idx, v in enumerate(seq):
        cur_time = clock()

        # skip first time
        if prev_time is not None:
            if print_iterations:
                print_iter(idx, cur_time - prev_time)

            times.append(cur_time - prev_time)

        prev_time = cur_time
        yield v

    # idx of last iteration
    idx += 1

    cur_time = clock()
    if print_iterations:
        print_iter(idx, cur_time - prev_time)

    times.append(cur_time - prev_time)
    times = np.array(times)

    total_iterations = idx
    total_time = np.sum(times)
    avg        = np.mean(times)
    sd         = np.std(times)
    median     = np.median(times)
    M          = np.max(times)
    m          = np.min(times)

    logger.log(log_level, "Performed {} in {} iterations. Time = {:.3f} s".format(title, total_iterations, total_time))
    logger.log(log_level, "   max = {:.3f} s".format(M))
    logger.log(log_level, "   avg = {:.3f} s".format(avg))
    logger.log(log_level, "   med = {:.3f} s".format(median))
    logger.log(log_level, "   min = {:.3f} s".format(m))
    logger.log(log_level, "   sd  = {:.3f} s".format(sd))

    return


def time(log_level=logging.DEBUG, title=None):
    """
    Outputs the time a function takes to execute.
    """
    def decorator(fn):
        name = title if title is not None else fn.__name__

        def wrapper(*args, **kwargs):
            module_logger.log(log_level, "{} started".format(name))

            t1 = clock()
            result = fn(*args, **kwargs)
            t2 = clock()

            module_logger.log(log_level, "{} finished in {:.4} ms".format(name, t2 - t1))
            return result

        return wrapper

    return decorator
