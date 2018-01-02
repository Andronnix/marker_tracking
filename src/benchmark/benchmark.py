from datetime import datetime
from typing import TextIO

import cv2
import os

import fern
import logging
import matplotlib
matplotlib.use("macosx")
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


@util.time(log_level=logging.INFO, title="Measuring dataset")
def measure_dataset(detector,
                    video,
                    frame_flags: TextIO,
                    gt_homography: TextIO,
                    gt_points: TextIO,
                    sample=None,
                    explore=False):
    if explore:
        logger.debug("Explore enabled")

    next(frame_flags), next(gt_homography), next(gt_points)

    h, w = np.shape(sample)[:2]
    sample_bounds = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

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

        points, H = detector.detect(frame, orig_bounds=sample_bounds)
        metric = calc_metric(truth, points)

        if explore:
            util.examine_detection(detector, sample, frame, truth, points)

        logger.debug("Metric value for frame {} = {}".format(idx, metric))
        result.append(metric)

    return result


@util.time(log_level=logging.INFO, title="Training detector")
def train_detector(video, gt_points: TextIO):
    assert video.isOpened()
    frame = next(util.get_frames(video))

    gt_points = np.array(list(util.grouper(map(float, next(gt_points).strip().split()), 2)))
    sample_corners = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=np.float32)

    H, _ = cv2.findHomography(gt_points, sample_corners, cv2.RANSAC, 5.0)
    sample = cv2.warpPerspective(frame, H, (640, 480))

    detector = fern.FernDetector.train(sample,
                                       deform_param_gen=util.smart_deformations_gen(sample, 20, ),
                                       max_train_corners=250,
                                       max_match_corners=500)
    return sample, detector


def plot_result(result, name):
    def count(t):
        return len(list(filter(lambda x: x <= t, result))) / len(result)

    X = list(range(100))
    precision = [count(threshold) for threshold in X]

    plt.plot(X, precision, label="{}, {:.3}".format(name, precision[5]))


def benchmark(ds_name):
    logger.info("Benchmarking dataset {}".format(ds_name))
    annotation_path = "datasets/annotation"
    video_path = "datasets/{}".format(ds_name)

    for fname in os.listdir("datasets/{}".format(ds_name)):
        logger.info("Using video {}".format(fname))

        vname = fname.rstrip(".avi")
        flagname = "{}_flag.txt".format(vname)
        Hname = "{}_gt_homography.txt".format(vname)
        ptsname = "{}_gt_points.txt".format(vname)

        logger.debug("Open files {}, {}, {}".format(flagname, Hname, ptsname))
        with open("{}/{}".format(annotation_path, flagname), 'r') as flag, \
             open("{}/{}".format(annotation_path, Hname),'r') as homography, \
             open("{}/{}".format(annotation_path, ptsname), 'r') as points:
            logger.debug("Open {}".format(fname))

            video = cv2.VideoCapture("{}/{}".format(video_path, fname))

            sample, detector = train_detector(video, points)

            logger.debug("Reset video an points file positions to start")
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            points.seek(0)

            result = measure_dataset(
                detector=detector,
                video=video,
                frame_flags=flag,
                gt_homography=homography,
                gt_points=points,
                sample=sample,
                explore=False)

            logger.info("Printing result")
            logger.info(result)

            logger.info("Plotting result")
            plot_result(result, vname)

    plt.title(ds_name)
    plt.xlabel("Alignment error threshold")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("log/plot_{}_{}.png".format(ds_name, START_TIME))
    plt.close()


if __name__ == "__main__":
    benchmark("V01")
    benchmark("V03")
    benchmark("V07")
    benchmark("V22")
