from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2
import serial

spfl = "shape_predictor_68_face_landmarks.dat"

COUNTER = 0
TOTAL = 0


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def blink(rect):
    global COUNTER
    global TOTAL
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    print("EAR: {0}".format(ear))
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    if ear < EYE_AR_THRESH:
        print("True")
        COUNTER += 1
        print("COUNTER: {0}".format(COUNTER))
    else:
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            TOTAL += 1
        print("TOTAL: {0}".format(TOTAL))
        COUNTER = 0
    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return leftEye


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(spfl)

eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

video_capture = cv2.VideoCapture(2)

# port = '/dev/ttyACM0'
# baud = 9600
# ser = serial.Serial(port, baud, timeout=6)
#
# print(ser.is_open)
#
# print("Stopping Boat")
# ser.write(b"S")
# a = ser.readline()
# print(a)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        leftEye = blink(rect)
        print(TOTAL)
        print(leftEye)

        px = 20

        x0 = leftEye[0][0] - px
        y0 = leftEye[0][1] - px
        print("X0 {0}".format(x0))
        print("Y0 {0}".format(y0))

        x1 = leftEye[3][0] + px
        y1 = leftEye[3][1] + px
        print("X1 {0}".format(x1))
        print("Y1 {0}".format(y1))

        eye = frame[y0:y1, x0:x1]
        eye = np.array(eye)

        # gleft = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        # circles = cv2.HoughCircles(gleft, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        # circles = np.uint16(np.around(circles))
        # print(len(circles))
        # n = 0
        # for i in circles[0, :]:
        #     if(i[0] == 0):
        #         n = 1
        #         print("Not Detected!!!")
        #         continue
        #     # draw the outer circle
        #     cv2.circle(eye, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #     # draw the center of the circle
        #     cv2.circle(eye, (i[0], i[1]), 2, (0, 0, 255), 3)
        #     print(i[0], i[1])
        #     center = [i[0] + x0, i[1] + y0]
        # if(n == 1):
        #     continue
        # cv2.imshow("gleft", gleft)
        cv2.imshow('eye2', eye)
        # print("Params")
        # print(leftEye[0])
        # print(leftEye[3])
        # print(center)
        # c = dist.euclidean(leftEye[0], leftEye[3])
        # d = dist.euclidean(leftEye[0], center)
        # e = dist.euclidean(center, leftEye[3])
        # print(c)
        # print(d)
        # print(e)
        #
        # l = 0.00
        # u = 15.00
        # l = np.float64(l)
        # u = np.float64(u)
        #
        # if ((d >= l) & (d <= u)):
        #     print("Left")
        #     # ser.write(b"L")
        #     # a = ser.readline()
        #     # print(a)
        #
        # if ((e >= l) & (e <= u)):
        #     print("Right")
        #     # ser.write(b"R")
        #     # a = ser.readline()
        #     # print(a)
        #
        # if ((d > u) & (e > u)):
        #     print("Center")
        #     # ser.write(b"F")
        #     # a = ser.readline()
        #     # print(a)
        #
        # cv2.imshow("gleft", gleft)
        # cv2.imshow('eye2', eye)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

ser.close()
cv2.destroyAllWindows()