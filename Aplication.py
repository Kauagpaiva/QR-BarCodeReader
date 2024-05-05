from ultralytics import YOLO
import cv2

# Importing the model
model = YOLO("../runs/detect/train8/weights/last.pt")

# Starting the camera
cap = cv2.VideoCapture(0)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

while cap.isOpened():

    # Gets the frames from the camera
    ret, frame = cap.read()

    if not ret:
        break

    if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # gets the detections from the frame
    results = model(frame)

    for data in results:
        if data:
            boxes = data.boxes.xyxy.tolist()
            #data.show()
            
            for box in boxes:
                # Extraindo coordenadas da bounding box
                x1, y1 = int(box[0]), int(box[1])
                x2, y2 = int(box[2]), int(box[3])

                # Desenhando a bounding box na imagem
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        else:
            pass
    cv2.imshow('Processando', frame)

cap.release()
cap.closeAlWindows()