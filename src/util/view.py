import cv2
import logging
import numpy as np
import util

COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_MAGENTA = (255, 0, 255)

module_logger = logging.getLogger("app.view")

def capture_img(cam, size=(640, 480)):
    _, img = cam.read()
    return cv2.resize(img, size)


def capture_orig(cam):
    while True:
        img = capture_img(cam)

        cv2.imshow("Press enter to capture", cv2.flip(img, 1))
        if cv2.waitKey(1) == 13:
            return img


def key_pressed(key=None, wait=1):
    pressed = cv2.waitKey(wait)
    if key is None:
        return 0 <= pressed < 255
    else:
        return pressed == key


def wait_for_key(key=None):
    module_logger.info("Waiting for key...")
    while not key_pressed(key):
        pass
    module_logger.info("Continuing")


def draw_poly(img, corners, color=COLOR_WHITE):
    corners = [np.int32(corners)]
    cv2.polylines(img, corners, True, color)


def explore_match(sample, match, kp_pairs, window_name="Match exploration", status=None, H=None):
    h1, w1 = sample.shape[:2]
    h2, w2 = match.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
    vis[:h1, :w1, :] = sample
    vis[:h2, w1:w1+w2, :] = match

    sample_copy = sample.copy()
    match_copy = match.copy()

    if H is not None:
        corners = [[0, 0], [w1, 0], [w1, h1], [0, h1]]
        corners = util.transform32(corners, H, (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0]))
        p2.append(np.int32(np.array(kpp[1]) + [w1, 0]))

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier[0]:
            col = COLOR_GREEN
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)

            cv2.circle(sample_copy, (x1, y1), 2, col, -1)
            cv2.circle(match_copy, (x2-w1, y2), 2, col, -1)
        else:
            col = COLOR_RED
            r = 2
            thickness = 1
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)

            cv2.circle(sample_copy, (x1, y1), 2, col, -1)
            cv2.circle(match_copy, (x2-w1, y2), 2, col, -1)

    # for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
    #     if not inlier[0]:
    #         cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 128))

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier[0]:
            cv2.line(vis, (x1, y1), (x2, y2), COLOR_GREEN)

    cv2.imshow(window_name, vis)
    cv2.imshow(window_name + " 1", sample_copy)
    cv2.imshow(window_name + " 2", match_copy)


    return vis


def anorm2(a):
    return (a * a).sum(-1)


def anorm(a):
    return np.sqrt(anorm2(a))


def explore_match_mouse(img1, img2, kp_t, kp_m, win_name="Match exploration", status=None, H=None):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    vis0 = vis.copy()

    if H is not None:
        corners0 = [[0, 0], [w1, 0], [w1, h1], [0, h1]]
        corners = util.transform32(corners0, H, (w1, 0))
        cv2.polylines(vis, [corners], True, util.COLOR_WHITE)
        for (x, y), (x1, y1) in zip(corners, corners0):
            cv2.line(vis, (x, y), (x1, y1), COLOR_WHITE, 1)

    if status is None:
        status = np.ones(len(kp_t), np.bool_)

    p1, p2 = kp_t, kp_m + (w1, 0)
    # for kpp in kp_pairs:
    #     p1.append(np.int32(kpp[0]))
    #     p2.append(np.int32(np.array(kpp[1]) + [0, w1]))

    p3 = util.transform32(p1, H, (w1, 0))
    p1, p2 = np.int32(p1), np.int32(p2)

    for (x1, y1), (x2, y2), (x3, y3), inlier in zip(p1, p2, p3, status):
        if inlier:
            col = COLOR_GREEN
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
            cv2.circle(vis, (x3, y3), 2, COLOR_MAGENTA, -1)
        else:
            col = COLOR_RED
            r = 2
            thickness = 1
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)

    for (x1, y1), (x2, y2), (x3, y3), inlier in zip(p1, p2, p3, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), COLOR_GREEN)
            cv2.line(vis, (x2, y2), (x3, y3), COLOR_MAGENTA)

    cv2.imshow(win_name, vis)

    def onmouse(event, x, y, flags, param):
        if not (flags & cv2.EVENT_FLAG_LBUTTON):
            return

        mouse_pos = (x, y)

        cur_vis = vis0.copy()

        if H is not None:
            cv2.polylines(cur_vis, [corners], True, (255, 255, 255))

        r = 16
        m = (anorm(np.array(p1) - mouse_pos) < r) | (anorm(np.array(p2) - mouse_pos) < r)
        idxs = np.where(m)[0]

        for i in idxs:
            (x1, y1), (x2, y2), (x3, y3) = p1[i], p2[i], p3[i]
            col = COLOR_GREEN if status[i] else COLOR_RED
            cv2.line(cur_vis, (x1, y1), (x2, y2), col)
            cv2.line(cur_vis, (x2, y2), (x3, y3), COLOR_MAGENTA)

        cv2.imshow(win_name, cur_vis)

    cv2.setMouseCallback(win_name, onmouse)
    return vis


def examine_detection(detector, sample, img, truth_box, detection_box, explore=True):
    img2 = img.copy()
    util.draw_poly(img2, truth_box, color=util.COLOR_WHITE)
    util.draw_poly(img2, detection_box, color=util.COLOR_RED)

    cv2.imshow("step", img2)

    if explore:
        kp_t, kp_m, kp_p = detector.match(img)
        H, status = cv2.findHomography(np.array(kp_t), np.array(kp_m), cv2.RANSAC, 5.0)
        if H is not None:
            explore_match_mouse(sample, img, kp_t, kp_m, H=H, status=status)
        else:
            module_logger.error("Homography not found")

    util.wait_for_key()