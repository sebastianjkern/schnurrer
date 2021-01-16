import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img = cv2.imread("face.jpg")
schnurrbart = cv2.imread("schnurrbart.png")

cv2.imshow(winname="Face", mat=schnurrbart)

cv2.waitKey(delay=0)

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

    # Calculate the bounding box
    max_x = 0
    max_y = 0
    min_x = 0
    min_y = 0

    if top_left_x > top_right_x:
        max_x = top_left_x
    else:
        max_x = top_right_x

    if bottom_left_x < bottom_right_x:
        min_x = bottom_left_x
    else:
        min_x = bottom_right_x

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

    cv2.polylines(img, box, True, (0, 255, 0), thickness=3)

    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y

        cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

cv2.imshow(winname="Face", mat=img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()
