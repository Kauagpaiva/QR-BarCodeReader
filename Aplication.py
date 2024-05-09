import cv2
from pyzbar.pyzbar import decode #QR/Bar code reader
import numpy as np

# Starting the camera
cap = cv2.VideoCapture(0)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

while cap.isOpened():

    # Gets the frames from the camera
    sucess, frame = cap.read()

    if not sucess:
        break

    if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    for code in decode(frame):
        points = np.array([code.polygon], np.int32)
        points = points.reshape((-1,1,2))
        cv2.polylines(frame,[points],True, (0,250,0),5)

        data = code.data.decode('utf-8')
        points2 = code.rect
        cv2.putText(frame, data,(points2[0], points2[1]),cv2.FONT_HERSHEY_COMPLEX, 1, (0,250,0), 2)

    if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.imshow('Processing', frame)

cap.release()
cv2.destroyAllWindows()