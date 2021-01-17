import cv2
import dlib
import numpy as np
import sys

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img = cv2.imread(sys.argv[1])

if img is None:
    exit()

dimensions = img.shape

schnurrbart = cv2.imread("schnurrbart.jpg")
schnurrbart = cv2.resize(schnurrbart, (dimensions[0], dimensions[1]))
schnurrbart = (255 - schnurrbart)

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

    m_mouth = (mouth_left_y - mouth_right_y) / (mouth_left_x - mouth_right_x)
    n_mouth = mouth_point_y - m_mouth * mouth_point_x

    m_nose = (nose_left_y - nose_right_y) / (nose_left_x - nose_right_x)
    n_nose = nose_point_y - m_nose * nose_point_x

    # Calculate the four points
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

    # Calculate the bounding box, Probably not needed
    max_x = 0
    max_y = 0
    min_x = 0
    min_y = 0

    if bottom_right_x > top_right_x:
        max_x = bottom_right_x
    else:
        max_x = top_right_x

    if bottom_left_x < top_left_x:
        min_x = bottom_left_x
    else:
        min_x = top_left_x

    if top_left_y > top_right_y:
        max_y = top_left_y
    else:
        max_y = top_right_y

    if bottom_left_y < bottom_right_y:
        min_y = bottom_left_y
    else:
        min_y = bottom_right_y

    box = np.array([[[top_left_x, top_left_y], [top_right_x, top_right_y], [bottom_right_x, bottom_right_y],
                     [bottom_left_x, bottom_left_y]]], np.int32)

    bounding_box = np.array([[[max_x, min_y], [max_x, max_y], [min_x, max_y], [min_x, min_y]]], np.int32)

    # FIXME: Orientation issues
    sm_area_box = cv2.minAreaRect(box)
    points = cv2.boxPoints(sm_area_box)
    points = np.int0(points)

    src = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])
    int_src = np.int0(src)
    dst = np.float32(points)
    perspective_transform_matrix = cv2.getPerspectiveTransform(src, dst)

    transformed_schnurrbart = cv2.warpPerspective(schnurrbart, perspective_transform_matrix,
                                                  (img.shape[1], img.shape[0]))
    cv2.threshold(transformed_schnurrbart, 10, 255, type=0, dst=transformed_schnurrbart)

    mask = cv2.cvtColor(src=transformed_schnurrbart, code=cv2.COLOR_BGR2GRAY)
    transformed_schnurrbart = (255 - transformed_schnurrbart)

    cv2.copyTo(transformed_schnurrbart, mask, img)

cv2.imshow(winname="Face", mat=img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()

cv2.imwrite("faceWithSchnurrbart.jpg", img)