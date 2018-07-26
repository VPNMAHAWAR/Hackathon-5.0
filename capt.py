import cv2

video_capture = cv2.VideoCapture(2)
i = 1
while True:
    ret, frame = video_capture.read()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
       img = "Capt/Image"
       img += str(i)
       img += ".jpg"
       cv2.imwrite(img, frame)
       i += 1
    elif key == ord("q"):
        break

cv2.destroyAllWindows()