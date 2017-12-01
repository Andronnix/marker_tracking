from datetime import datetime
from typing import TextIO

import cv2
import fern
import logging
import matplotlib.pyplot as plt
import numpy as np
import util

START_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler("log/bench_{}.log".format(START_TIME))
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s %(name)-25s %(levelname)-8s %(message)s'))

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(name)-25s %(levelname)-8s %(message)s'))

logger.addHandler(fh)
logger.addHandler(ch)

# now we have proper logger
logger = logging.getLogger("app.benchmark")


def calc_metric(orig, points):
    """
    See Planar Object Tracking in the Wild A Benchmark, p. 6
    :param orig:
    :param points:
    :return:
    """
    result = 0
    for (xo, yo), (xp, yp) in zip(orig, points):
        result += (xo - xp) ** 2 + (yo - yp) ** 2

    return np.sqrt(result) / 2


def benchmark_dataset(detector,
                      video,
                      frame_flags: TextIO,
                      gt_homography: TextIO,
                      gt_points: TextIO,
                      sample=None,
                      explore=False):
    logger.debug("Benchmarking dataset")
    if explore:
        logger.debug("Explore enabled")

    logger.debug("Drop first line of data")
    next(frame_flags), next(gt_homography), next(gt_points)

    logger.debug("Start iterating over frames")
    result = []
    for idx, (frame, flag, Hline, Pline) in \
            enumerate(zip(util.get_frames(video), frame_flags, gt_homography, gt_points)):
        logger.debug("Evaluating frame {}".format(idx))
        truth = list(util.grouper(map(float, Pline.strip().split()), 2))
        flag = int(flag.strip())

        if idx % 2 == 1 or flag > 0:
            logger.debug("Frame {} dropped".format(idx))
            continue

        points, H = detector.detect(frame)
        metric = calc_metric(truth, points)

        if explore:
            util.examine_detection(detector, sample, frame, truth, points)

        logger.debug("Metric value for frame {} = {}".format(idx, metric))
        result.append(metric)

    logger.debug("Benchmarking dataset done")
    return result


def train_detector(video, gt_points: TextIO):
    logger.info("Start detector training")
    frame = next(util.get_frames(video))

    gt_points = np.array(list(util.grouper(map(float, next(gt_points).strip().split()), 2)))
    sample_corners = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=np.float32)

    H, _ = cv2.findHomography(gt_points, sample_corners, cv2.RANSAC, 5.0)
    sample = cv2.warpPerspective(frame, H, (640, 480))

    detector = fern.FernDetector.train(sample, max_train_corners=50, max_match_corners=500)
    logger.info("Detector trained")
    return sample, detector


def plot_result(result):
    def count(t):
        return len(list(filter(lambda x: x <= t, result))) / len(result)

    X = list(range(1000))
    precision = [count(threshold) for threshold in X]

    plt.plot(X, precision)
    plt.xlabel("Alignment error threshold")
    plt.ylabel("Precision")
    plt.savefig("log/plot_{}.png".format(START_TIME))


if __name__ == "__main__":
    logger.info("Benchmark started")

    logger.debug("Open files V01_1_flag.txt, V01_1_gt_homography.txt, V01_1_gt_points.txt")
    with open("../../datasets/annotation/V01_1_flag.txt", 'r') as flag, \
         open("../../datasets/annotation/V01_1_gt_homography.txt", 'r') as homography, \
         open("../../datasets/annotation/V01_1_gt_points.txt", 'r') as points:

        logger.debug("Open V01_1.avi")
        video = cv2.VideoCapture("../../datasets/V01/V01_1.avi")

        sample, detector = train_detector(video, points)

        logger.debug("Reset video an points file positions to start")
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        points.seek(0)

        result = benchmark_dataset(
            detector=detector,
            video=video,
            frame_flags=flag,
            gt_homography=homography,
            gt_points=points,
            sample=sample,
            explore=True)

        logger.info("Printing result")
        logger.info(result)

        logger.info("Plotting result")
        plot_result(result)
