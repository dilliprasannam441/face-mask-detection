import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load face detection model (OpenCV DNN)
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
face_net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load mask detection model
model = load_model("model/mask_detector.h5")

# Start video capture
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare image for DNN
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # Loop through detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # adjust threshold if needed
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Ensure box is within image dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (224, 224))
            face_array = np.expand_dims(face_resized, axis=0) / 255.0
            (pred_mask, pred_without_mask) = model.predict(face_array)[0]

            if pred_mask > pred_without_mask:
                label = f"Masked ({int(pred_mask * 100)}%)"
                color = (0, 255, 0)
            else:
                label = f"Unmasked ({int(pred_without_mask * 100)}%)"
                color = (0, 0, 255)

            

            # Display label and box
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
