import cv2
import fern
import logging
import util

logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(name)-25s %(levelname)-8s %(message)s'))
logger.addHandler(ch)

# get proper logger
logger = logging.getLogger("app.examine")


def train_and_serialize(sample, serialization_path):
    detector = fern.FernDetector.train(sample, max_train_corners=50, max_match_corners=500)

    with open(serialization_path, 'w') as f:
        detector.serialize(f)

    detection_box, _ = detector.detect(sample)
    if len(detection_box) == 0:
        print("Homography not found")

    util.examine_detection(detector, sample, sample, [], detection_box, explore=True)


if __name__ == "__main__":
    sample = cv2.imread("samples/sample_ricotta.jpg")
    serialization_path = "samples/ricotta_detector.dat"
    train_and_serialize(sample, serialization_path)
