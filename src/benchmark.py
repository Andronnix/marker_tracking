import cv2
import fern
import numpy as np
import os
import pickle
import random
import util


def benchmark_dataset(dataset, explore=True):
    ground_truth = util.get_ground_truth(dataset)
    sample = util.get_sample(dataset)

    detector = fern.FernDetector(sample, max_train_corners=20, max_match_corners=500)
    detector.draw_learned_ferns()
    # detector.draw_learned_ferns_2("img/learn/")

    img = sample.copy()
    detection_box, _ = detector.detect(img)
    examine_detection(detector, sample, img, ground_truth[0], detection_box, explore=explore)

    for truth_box, img in zip(ground_truth, util.get_images(dataset)):
        detection_box, _ = detector.detect(img)
        if len(detection_box) == 0:
            print("Homography not found")

        examine_detection(detector, sample, img, truth_box, detection_box, explore=explore)


def benchmark_sample(deserialize=False):
    cam = cv2.VideoCapture("samples/test_ricotta.avi")
    sample = cv2.imread("samples/sample_ricotta.jpg")

    serialization_path = "samples/ricotta_detector.dat"

    if not deserialize:
        detector = fern.FernDetector.train(sample, max_train_corners=50, max_match_corners=500)

        with open(serialization_path, 'w') as f:
            detector.serialize(f)

    else:
        with open(serialization_path, 'r') as f:
            detector = fern.FernDetector.deserialize(f)

    detector.draw_learned_ferns()

    detection_box, _ = detector.detect(sample)
    if len(detection_box) == 0:
        print("Homography not found")

    # examine_detection(detector, sample, sample, [], detection_box, explore=True)

    while True:
        ret, img = cam.read()
        if not ret:
            break

        detection_box, _ = detector.detect(img)
        if len(detection_box) == 0:
            print("Homography not found")

        examine_detection(detector, sample, img, [], detection_box, explore=True)

    cam.release()


def examine_detection(detector, sample, img, truth_box, detection_box, explore=True):
    img2 = img.copy()
    util.draw_poly(img2, truth_box, color=util.COLOR_WHITE)
    util.draw_poly(img2, detection_box, color=util.COLOR_RED)

    cv2.imshow("step", img2)

    if explore:
        kp_t, kp_m, kp_p = detector.match(img)
        H, status = cv2.findHomography(np.array(kp_t), np.array(kp_m), cv2.RANSAC, 5.0)
        util.explore_match_mouse(sample, img, kp_t, kp_m, H=H, status=status)
        #util.explore_match(sample, img, kp_pairs=kp_p, H=H, status=status)

    util.wait_for_key()


if __name__ == "__main__":
    random.seed(1234)
    # benchmark_sample()
    benchmark_sample(deserialize=True)
    # benchmark_dataset("ClifBar")
    # benchmark_dataset("Box")
