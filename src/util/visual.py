from collections import namedtuple
from .timers import iter_timer, Timer
from .other import flip_points, mult

import cv2
import logging
import numpy as np

logger = logging.getLogger("app.visual")


def get_corners(img, max_corners):
    corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=0.01, minDistance=8)
    return ((y, x) for ((x, y),) in corners)


Collector = namedtuple("Collector", ["x", "y", "cnt"])


def get_stable_corners(train_img, max_corners=100):
    logger.debug("Generating {} stable corners".format(max_corners))
    H, W = np.shape(train_img)[:2]

    CORNER_CNT = 500

    corners = list(get_corners(train_img, CORNER_CNT))

    with Timer("Generating deformed images and collect corers"):
        generator = default_deformations_gen((W / 2, H / 2), theta_step=12, deformations=5)

        for _, R_inv, img in generate_deformations(train_img, generator):
            new_corners = np.array(list(get_corners(img, CORNER_CNT)), dtype=np.float32)

            # y,x --> x,y
            new_corners = flip_points(new_corners)

            t = [[1]] * len(new_corners)
            new_corners = np.transpose(np.hstack((new_corners, t)))
            corners_inv = np.transpose(np.dot(R_inv, new_corners))

            corners.extend(corners_inv)

    corners = sorted(corners, key=lambda p: p[0])

    collectors = []

    def find_best_collector(x, y):
        threshold = 2 ** 2

        for idx in reversed(range(len(collectors))):
            c = collectors[idx]

            xdist = abs(c.x - x)
            dist2 = xdist ** 2 + (c.y - y) ** 2
            if dist2 <= threshold:
                return idx

            # collectors are sorted by x
            # when xdist > threshold then for all remaining collectors dist2 > threshold
            if xdist > threshold:
                break

        return None

    skip_count = 0
    for cx, cy in iter_timer(corners, "Detect stable corners", print_iterations=False):
        if not (0 <= cy < H and 0 <= cx < W):
            skip_count += 1
            continue

        best_collector_idx = find_best_collector(cx, cy)
        if best_collector_idx is None:
            collectors.append(Collector(cx, cy, 1))
        else:
            x, y, cnt = collectors[best_collector_idx]
            collectors[best_collector_idx] = Collector((x * cnt + cx) / (cnt + 1), (y * cnt + cy) / (cnt + 1), cnt + 1)

    collectors = sorted(collectors, key=lambda c: -c.cnt)

    logger.debug("Found {} stable corners. Yielding".format(min(max_corners, len(collectors))))
    for x, y, cnt in collectors[:max_corners]:
        yield int(y), int(x)


def default_deformations_gen(center, theta_step=60, deformations=20):
    rotation_matrices = [
        cv2.getRotationMatrix2D(center, theta, 1.0)
        for theta in range(0, 361)
    ]

    for theta in range(0, 360, theta_step):
        Rt = rotation_matrices[theta]
        r_phi = np.random.randint(0, 360, deformations)
        r_lambda1 = np.random.uniform(0.25, 1.5, deformations)
        r_lambda2 = np.random.uniform(0.25, 1.5, deformations)
        r_noise_ratio = np.random.uniform(0, 0.1, deformations)

        for noise_ratio, lambda1, lambda2, phi in zip(r_noise_ratio, r_lambda1, r_lambda2, r_phi):
            Rp = rotation_matrices[phi]
            Rp1 = rotation_matrices[360 - phi]
            yield Rt, noise_ratio, lambda1, lambda2, Rp, Rp1


def smart_deformations_gen(sample, theta_range, deformations=20):
    h, w = np.shape(sample)[:2]
    center = (w / 2.0, h / 2.0)

    rotation_matrices = [
        cv2.getRotationMatrix2D(center, theta, 1.0)
        for theta in range(0, 361)
    ]

    thetas = np.array(list(range(-theta_range, theta_range)))
    thetas += 360
    thetas %= 360

    for theta in thetas:
        Rt = rotation_matrices[theta]
        r_phi = np.random.randint(0, 360, deformations)
        r_lambda1 = np.random.uniform(0.25, 1.5, deformations)
        r_lambda2 = np.random.uniform(0.25, 1.5, deformations)
        r_noise_ratio = np.random.uniform(0, 0.1, deformations)

        for noise_ratio, lambda1, lambda2, phi in zip(r_noise_ratio, r_lambda1, r_lambda2, r_phi):
            Rp = rotation_matrices[phi]
            Rp1 = rotation_matrices[360 - phi]
            yield Rt, noise_ratio, lambda1, lambda2, Rp, Rp1


def generate_deformations(img, deform_param_gen=None):
    H, W = np.shape(img)[:2]
    center = np.float32(W / 2.0), np.float32(H / 2.0)

    if deform_param_gen is None:
        deform_param_gen = default_deformations_gen(center)

    for Rt, noise_ratio, lambda1, lambda2, Rp, Rp1 in deform_param_gen:
        Rl = np.matrix([[lambda1, 0, 0], [0, lambda2, 0]])
        Rz = mult(Rp, mult(Rl, Rp1))
        R = mult(Rt, Rz)
        R_inv = cv2.invertAffineTransform(R)

        warped = cv2.warpAffine(img, R, dsize=(W, H), borderMode=cv2.BORDER_REPLICATE)
        blurred = cv2.GaussianBlur(warped, (7, 7), 25)

        # add gaussian noise
        noise = np.uint8(np.random.normal(0, 25, (H, W)))
        noised = cv2.addWeighted(blurred, 1 - noise_ratio, noise, noise_ratio, 0)

        yield R, R_inv, noised


def generate_patch(img, center, size):
    h, w = np.shape(img)
    h, w = int(h), int(w)

    ph, pw = size

    assert 0 < pw <= w and 0 < ph <= h

    ph2, pw2 = ph // 2, pw // 2
    y, x = center
    y, x = int(y), int(x)

    if pw2 <= x <= w - pw2 and ph2 <= y <= h - ph2:
        # fast way
        return img[y - ph2:y + ph2, x - pw2:x + pw2]

    assert 0 <= y < h and 0 <= x < w, "Point outside of the image. (y, x)=({}, {}) (h, w)=({}, {})".format(y, x, h, w)

    top_adj = 0
    bottom_adj = 0
    y0 = y - ph2
    y1 = y0 + ph
    if y0 < 0:
        top_adj = -y0
        y0 = 0
        y1 += top_adj
    if y1 > h:
        bottom_adj = y1 - h

    left_adj = 0
    right_adj = 0
    x0 = x - pw2
    x1 = x0 + pw
    if x0 < 0:
        left_adj = -x0
        x0 = 0
        x1 += left_adj
    if x1 > w:
        right_adj = x1 - w

    img_extended = cv2.copyMakeBorder(img, top_adj, bottom_adj, left_adj, right_adj, cv2.BORDER_REFLECT101)
    result = img_extended[y0:y0 + ph, x0:x0 + pw]
    assert np.shape(result)[:2] == size, "Wrong size"
    return result


def generate_patch_class(img, corner, patch_size):
    """ generate patch transformations """

    patch = generate_patch(img, corner, np.array(patch_size) * 2)
    for _, _, img in generate_deformations(patch, patch_size):
        yield img


def generate_key_point_pairs(patch_size, n=300):
    pw, ph = patch_size

    xs0 = np.random.random_integers(1, pw - 2, n)
    ys0 = np.random.random_integers(1, ph - 2, n)

    xs1 = np.random.random_integers(1, pw - 2, n)
    ys1 = np.random.random_integers(1, ph - 2, n)

    for x0, y0, x1, y1 in zip(xs0, ys0, xs1, ys1):
        yield (y0, x0), (y1, x1)


def get_frames(video):
    while True:
        frame_captured, frame = video.read()

        if not frame_captured:
            return

        yield frame
