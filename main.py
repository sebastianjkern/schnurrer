import sys

import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

if len(sys.argv) < 2:
    print("Usage: python main.py <path>")
    exit()

img = cv2.imread(sys.argv[1])

if img is None:
    print("Something went wrong trying to read the image...")
    exit()

dimensions = img.shape

moustache = cv2.imread("moustache.jpg")
moustache = cv2.resize(moustache, (dimensions[1], dimensions[0]))
moustache = (255 - moustache)

gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
faces = detector(gray)

for face in faces:
    landmarks = predictor(image=gray, box=face)

    mouth_point_x = landmarks.part(33).x
    mouth_point_y = landmarks.part(33).y

    mouth_left_x = landmarks.part(48).x
    mouth_left_y = landmarks.part(48).y

    mouth_right_x = landmarks.part(54).x
    mouth_right_y = landmarks.part(54).y

    nose_point_x = landmarks.part(51).x
    nose_point_y = landmarks.part(51).y

    nose_left_x = landmarks.part(31).x
    nose_left_y = landmarks.part(31).y

    nose_right_x = landmarks.part(35).x
    nose_right_y = landmarks.part(35).y

    print(landmarks.part(32))

    m_mouth = (mouth_left_y - mouth_right_y) / (mouth_left_x - mouth_right_x)
    n_mouth = mouth_point_y - m_mouth * mouth_point_x

    m_nose = (nose_left_y - nose_right_y) / (nose_left_x - nose_right_x)
    n_nose = nose_point_y - m_nose * nose_point_x

    m_left = (mouth_left_y - nose_left_y) / (mouth_left_x - nose_left_x)
    n_left = mouth_left_y - m_left * mouth_left_x

    m_right = (mouth_right_y - nose_right_y) / (mouth_right_x - nose_right_x)
    n_right = mouth_right_y - m_right * mouth_right_x

    top_left_x = (n_nose - n_left) / (m_left - m_nose)
    top_left_y = m_nose * top_left_x + n_nose

    top_right_x = (n_nose - n_right) / (m_right - m_nose)
    top_right_y = m_nose * top_right_x + n_nose

    bottom_left_x = (n_mouth - n_left) / (m_left - m_mouth)
    bottom_left_y = m_mouth * bottom_left_x + n_mouth

    bottom_right_x = (n_mouth - n_right) / (m_right - m_mouth)
    bottom_right_y = m_mouth * bottom_right_x + n_mouth

    box = np.array([[[bottom_right_x, bottom_right_y], [bottom_left_x, bottom_left_y], [top_left_x, top_left_y],
                     [top_right_x, top_right_y]]], np.int32)

    src = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])
    dst = np.float32(box)
    perspective_transform_matrix = cv2.getPerspectiveTransform(src, dst)

    transformed_moustache = cv2.warpPerspective(moustache, perspective_transform_matrix,
                                                (img.shape[1], img.shape[0]))
    cv2.threshold(transformed_moustache, 10, 255, type=0, dst=transformed_moustache)

    mask = cv2.cvtColor(src=transformed_moustache, code=cv2.COLOR_BGR2GRAY)
    transformed_moustache = (255 - transformed_moustache)

    cv2.copyTo(transformed_moustache, mask, img)

cv2.imshow(winname="Face", mat=img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()

cv2.imwrite("modified_face.jpg", img)
