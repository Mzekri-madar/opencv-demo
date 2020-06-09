import numpy as np
import cv2
import logging as log
import datetime as dt

cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
log.basicConfig(filename="webcam.log", level=log.INFO)

cap = cv2.VideoCapture(0)
anterior = 0

while True:
    # Process frame
    ret, frame = cap.read()

    # Get gray frame
    gary = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gary, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

    # Display frame
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
