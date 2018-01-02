from typing import IO

import cv2
import logging
import numpy as np
import random
import util

from collections import defaultdict, namedtuple

module_logger = logging.getLogger("app.fern")


class Fern:
    def __init__(self, size, key_point_pairs):
        self._size = size
        self.kp_pairs = key_point_pairs

    def calculate(self, sample):
        """ Calculates feature function on all kp_pairs and generates binary number according to article"""
        result = 0

        for (y1, x1), (y2, x2) in self.kp_pairs:
            result *= 2
            result += 0 if (sample[y1, x1] < sample[y2, x2]) else 1

        return result

    def draw(self, k, sample):
        levels = []

        for _ in range(len(self.kp_pairs)):
            levels.append(k % 2)
            k >>= 1

        levels.reverse()
        for ((y1, x1), (y2, x2)), level in zip(self.kp_pairs, levels):
            sample[y1, x1] = 255 * level
            sample[y2, x2] = 255 * (1 - level)

    def serialize(self, file: IO):
        file.write("{},{}\n".format(
            len(self.kp_pairs),
            ",".join(util.flatmap2(str, self.kp_pairs))
        ))

    @staticmethod
    def deserialize(file: IO):
        cnt, *points = file.readline().strip().split(",")
        cnt = int(cnt)
        points = list(map(int, points))
        assert len(points) == cnt * 4, "Can't deserialize Fern. count = {}, coords = {}"

        return Fern(cnt, list(util.grouper(util.grouper(points, 2), 2)))


KPMatch = namedtuple("KPMatch", ["val", "point"])


class FernDetector:
    @staticmethod
    def train(sample, deform_param_gen=None, patch_size=(32, 32), max_train_corners=40, max_match_corners=200):
        module_logger.info("Training FernDetector")
        fd = FernDetector(patch_size=patch_size,
                          max_train_corners=max_train_corners,
                          max_match_corners=max_match_corners)
        fd._init_ferns()
        fd._train(sample, deform_param_gen)
        module_logger.info("FernDetector trained")
        return fd

    def __init__(self,
                 patch_size=(16, 16),
                 max_train_corners=40,
                 max_match_corners=200,
                 ferns=None,
                 ferns_p=None,
                 classes_cnt=1,
                 key_points=None,
                 fern_bits=None):
        self._patch_size = patch_size
        self._max_train_corners = max_train_corners
        self._max_match_corners = max_match_corners
        self._ferns = ferns
        self._fern_p = ferns_p
        self._classes_count = classes_cnt
        self._fern_bits = fern_bits
        self.key_points = key_points
        self.logger = logging.getLogger("app.fern.FernDetector")

    _K = property(lambda self: 2 ** (self._fern_bits + 1))

    @util.time(log_level=logging.INFO, title="Initializing ferns")
    def _init_ferns(self, fern_bits=11, fern_count=15):
        self.logger.debug("Init params: fern_bits={}, fern_count={}".format(fern_bits, fern_count))

        self._fern_bits = fern_bits
        kp_pairs = list(util.generate_key_point_pairs(self._patch_size, n=fern_bits*fern_count))

        # maps key_point[i] to fern[fern_indices[i]]
        fern_indices = []
        for fern_index in range(fern_count):
            fern_indices += [fern_index] * fern_bits

        random.shuffle(fern_indices)

        fern_kp_pairs = defaultdict(list)
        for kp_idx, fern_idx in enumerate(fern_indices):
            fern_kp_pairs[fern_idx].append(kp_pairs[kp_idx])

        self._ferns = [Fern(self._patch_size, kp_pairs) for fern_idx, kp_pairs in fern_kp_pairs.items()]

    @util.time(log_level=logging.INFO, title="Training ferns")
    def _train(self, train_img, deform_param_gen):
        img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        H, W = np.shape(img_gray)[:2]
        self.logger.debug("Training image size (w, h) = ({}, {})".format(W, H))

        corners = list(util.get_stable_corners(img_gray, self._max_train_corners))

        cp = train_img.copy()
        for y, x in corners:
            cv2.circle(cp, (x, y), 3, util.COLOR_WHITE, 3)
        cv2.imshow("Stable corners", cp)
        cv2.waitKey(10)

        self._classes_count = len(corners)
        self.logger.debug("Allocating probability matrix: ferns x classes x K = {} x {} x {}".format(
            len(self._ferns), self._classes_count, self._K
        ))
        self._fern_p = np.zeros((len(self._ferns), self._classes_count, self._K))
        self.key_points = []

        skipped = 0
        train_patches = 0
        title = "Training {} classes".format(self._classes_count)
        for R, _, img in util.iter_timer(util.generate_deformations(img_gray, deform_param_gen), title, False):
            new_corners = util.flip_points(corners)
            t = [[1]] * len(new_corners)
            new_corners = np.transpose(np.hstack((new_corners, t)))
            deformed_corners = util.flip_points(np.asarray(np.transpose(np.dot(R, new_corners))))

            for class_idx, (corner, deformed_corner) in enumerate(zip(corners, deformed_corners)):
                self.key_points.append(corner)

                cy, cx = deformed_corner
                if not (0 <= cy <= H and 0 <= cx <= W):
                    skipped += 1
                    continue

                train_patches += 1

                patch = util.generate_patch(img, deformed_corner, self._patch_size)

                for fern_idx, fern in enumerate(self._ferns):
                    k = fern.calculate(patch)
                    assert 0 <= k < self._K, "WTF!!!"
                    self._fern_p[fern_idx, class_idx, k] += 1
        self.logger.debug("skipped {} / {} deformations".format(skipped, train_patches))

        Nr = 1
        for fern_idx in util.iter_timer(range(len(self._ferns)), title="Calculating probs"):
            for cls_idx in range(self._classes_count):
                Nc = np.sum(self._fern_p[fern_idx, cls_idx, :])
                self._fern_p[fern_idx, cls_idx, :] += Nr
                self._fern_p[fern_idx, cls_idx, :] /= Nc + self._K * Nr

        self._fern_p = np.log(self._fern_p)

    @util.time(log_level=logging.INFO)
    def match(self, image):
        dims = len(np.shape(image))
        if dims == 3:
            self.logger.debug("Converting image to GRAY before matching")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        with util.Timer("extract corners"):
            corners = util.get_corners(image, self._max_match_corners)

        image = cv2.GaussianBlur(image, (7, 7), 25)

        EMPTY_VAL = -1000000
        best_match_val = np.zeros((self._classes_count,), dtype=np.float32)
        best_match_val += EMPTY_VAL
        best_match_corner = np.zeros((self._classes_count, 2), dtype=np.int32)
        for corner in util.iter_timer(corners, title="Matching corners", print_iterations=False):
            probs = np.zeros((self._classes_count,))

            patch = util.generate_patch(image, corner, self._patch_size)
            for fern_idx, fern in enumerate(self._ferns):
                k = fern.calculate(patch)
                probs += self._fern_p[fern_idx, :, k]

            most_probable_class = np.argmax(probs)
            most_prob = probs[most_probable_class]

            if most_prob > best_match_val[most_probable_class]:
                best_match_val[most_probable_class] = most_prob
                best_match_corner[most_probable_class] = corner

        key_points_trained = []
        key_points_matched = []
        key_points_pairs = []
        for cls in range(self._classes_count):
            if best_match_val[cls] == EMPTY_VAL:
                continue

            key_points_trained.append(self.key_points[cls])
            key_points_matched.append(best_match_corner[cls])
            key_points_pairs.append((self.key_points[cls], best_match_corner[cls]))

        return util.flip_points(key_points_trained), \
               util.flip_points(key_points_matched), \
               key_points_pairs

    @util.time(log_level=logging.INFO, title="Detecting")
    def detect(self, image, orig_bounds=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kp_t, kp_m, kpp = self.match(image)
        H, status = cv2.findHomography(kp_t, kp_m, cv2.RHO, 10.0)
        self.logger.debug("Found {} inliers out of {} pairs".format(sum(status), len(status)))

        if orig_bounds is None:
            h, w = np.shape(image)
            orig_bounds = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        if H is not None:
            self.logger.debug("Detection success")
            return util.transform32(orig_bounds, H), H

        self.logger.debug("Nothing detected")
        return [], H

    def _draw_patch_class(self, patches, cls_idx):
        w, h = self._patch_size

        W = (w + 5) * 10
        H = (h + 5) * (len(patches) // 10 + 1)

        img = np.zeros((W, H))
        for idx, patch in enumerate(patches):
            x = (idx // 10) * (w + 5)
            y = (idx % 10) * (h + 5)

            img[y:y + h, x: x + w] = patch

        cv2.imwrite("img/train/cls{}.png".format(cls_idx), img)

    @util.time(log_level=logging.INFO, title="Serialization")
    def serialize(self, file: IO):
        file.write("1\n")  # version
        file.write("{}\n".format(len(self._ferns)))
        file.write("{},{}\n".format(*self._patch_size))
        for fern in self._ferns:
            fern.serialize(file)

        F, C, K = np.shape(self._fern_p)

        file.write("{},{},{}\n".format(self._fern_bits, self._max_train_corners, self._max_match_corners))
        file.write("{},{},{}\n".format(F, C, K))

        for f in range(F):
            for c in range(C):
                file.write(
                    (",".join(map(str, self._fern_p[f, c, :]))) + "\n"
                )

        file.write(",".join(util.flatmap(str, self.key_points)) + "\n")

    @staticmethod
    @util.time(log_level=logging.INFO, title="Deserialization")
    def deserialize(file: IO):
        module_logger.info("Loading FernDetector from {}".format(file.name))
        version = int(file.readline().strip())

        if version != 1:
            msg = "Can't deserialize FernDetector from {}. Incorrect version of model. Expected 1, found {}"\
                .format(file.name, version)
            module_logger.error(msg)
            raise AssertionError(msg)

        num_ferns = int(file.readline().strip())
        ph, pw = map(int, file.readline().strip().split(","))

        with util.Timer("Deserializing ferns"):
            ferns = [Fern.deserialize(file) for _ in range(num_ferns)]

        fern_bits, max_train, max_match = map(int, file.readline().strip().split(","))

        with util.Timer("Deserializing fern_p"):
            F, C, K = map(int, file.readline().strip().split(","))
            fern_p = np.zeros((F, C, K), dtype=float)
            for fern_idx in range(F):
                for class_idx in range(C):
                    line = list(map(float, file.readline().strip().split(",")))
                    fern_p[fern_idx, class_idx, :] = line

        line = file.readline().strip().split(",")
        key_points = list(util.grouper(map(int, line), 2))

        module_logger.debug("Creating FernDetector")
        detector = FernDetector(
            patch_size=(ph, pw),
            max_train_corners=max_train,
            max_match_corners=max_match,
            ferns=ferns,
            ferns_p=fern_p,
            classes_cnt=C,
            key_points=key_points,
            fern_bits=fern_bits
        )

        return detector
