import os
import glob
import copy

import cv2
import numpy as np
import pandas as pd
import dlib

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Place for the pretrained model for dlib
PREDICTOR_PATH = f"{base_path}/weight/shape_predictor_68_face_landmarks.dat"

LEFT_INDEX = (36, 42)
RIGHT_INDEX = (42, 48)

LEFT_TOP_AREAS = [17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41]
RIGHT_TOP_AREAS = [22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47]
EYES_AREAS = LEFT_TOP_AREAS + RIGHT_TOP_AREAS
MOUTH_AREAS = list(range(29, 36)) + list(range(48, 60))

# Initialize the predictor and face detector
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def str_to_tuple(tuple_str: str):
    r"""Turn str tuple to real tuple
    For example: "(25, 32)" -> (25, 32)

    Parameters
    ----------
    tuple_str: str
        String to be changed

    Returns
    -------
    first_num: int
    second_num: int
    """
    first_num, second_num = tuple_str[1:-1].split(", ")
    return int(first_num), int(second_num)


def row_to_list(row):
    tuple_list = []

    for pair in row:
        tuple_list.append(str_to_tuple(pair))

    return tuple_list


def convert(landmarks):
    result = []
    for points in landmarks.parts():
        result.append((points.x, points.y))

    return result


def detect_landmarks(img):
    landmarks = predictor(img, dlib.rectangle(0, 0, img.shape[1], img.shape[0]))

    if landmarks is None:
        raise ValueError("landmarks is None")
    landmarks = convert(landmarks)

    return landmarks


def save_landmarks_csv(output_path):
    final_landmarks = []

    img_generator = glob.glob("*.jpg")
    for img_path in img_generator:
        landmarks = detect_landmarks(img_path)
        final_landmarks.append(landmarks)

    landmarks_csv = pd.DataFrame(final_landmarks)
    landmarks_csv.to_csv(output_path, index=False)


def eyes_move_box(boxes, shape, move=10):
    new_points = list(boxes)
    new_points[0] = max(new_points[0] - move, 0)
    new_points[1] = min(new_points[1] + move, shape[1])
    new_points[2] = max(new_points[2] - move, 0)
    new_points[3] = min(new_points[3] + move, shape[0])
    return new_points


def left_eye_move_box(boxes, shape, move=10):
    new_points = list(boxes)
    new_points[0] = max(new_points[0] - move, 0)
    new_points[2] = max(new_points[2] - move, 0)
    new_points[3] = new_points[3] + move
    return new_points


def right_eye_move_box(boxes, shape, move=10):
    new_points = list(boxes)
    new_points[1] = min(new_points[1] + move, shape[1])
    new_points[2] = max(new_points[2] - move, 0)
    new_points[3] = min(new_points[3] + move, shape[0])
    return new_points


def mouth_move_box(boxes, shape, move=10):
    new_points = list(boxes)
    new_points[0] = max(new_points[0] - move, 0)
    new_points[1] = min(new_points[1] + move, shape[1])
    new_points[2] = max(new_points[2] - move, 0)
    new_points[3] = min(new_points[3] + move, shape[0])
    return new_points


def find_boxes(points, shape):
    x_min = max(np.min(points[:, 0]), 0)
    x_max = min(np.max(points[:, 0]), shape[1])
    y_min = max(np.min(points[:, 1]), 0)
    y_max = min(np.max(points[:, 1]), shape[0])

    return x_min, x_max, y_min, y_max


def get_frame_from_box(img, box):
    return img[box[2] : box[3], box[0] : box[1]]


def get_left_right_mouth_img(img, move):
    face_landmarks = np.array(detect_landmarks(img))

    left_eye_boxes = left_eye_move_box(
        find_boxes(face_landmarks[LEFT_TOP_AREAS], img.shape[:2]), img.shape[:2], move
    )
    right_eye_boxes = right_eye_move_box(
        find_boxes(face_landmarks[RIGHT_TOP_AREAS], img.shape[:2]), img.shape[:2], move
    )
    mouth_boxes = mouth_move_box(
        find_boxes(face_landmarks[MOUTH_AREAS], img.shape[:2]), img.shape[:2], move
    )

    left_eye_img = get_frame_from_box(img, left_eye_boxes)
    right_eye_img = get_frame_from_box(img, right_eye_boxes)
    mouth_img = get_frame_from_box(img, mouth_boxes)

    return left_eye_img, right_eye_img, mouth_img


def get_eyes_mouth_img(img, move, target=None):
    if target is None:
        target = img

    face_landmarks = np.array(detect_landmarks(img))

    eyes_boxes = eyes_move_box(
        find_boxes(face_landmarks[EYES_AREAS], img.shape[:2]), img.shape[:2], move[0]
    )
    mouth_boxes = mouth_move_box(
        find_boxes(face_landmarks[MOUTH_AREAS], img.shape[:2]), img.shape[:2], move[1]
    )

    eyes_img = get_frame_from_box(target, eyes_boxes)
    mouth_img = get_frame_from_box(target, mouth_boxes)

    return (eyes_img, mouth_img), (eyes_boxes, mouth_boxes)


if __name__ == "__main__":
    pass
