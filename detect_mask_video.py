# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy  # Import for loss computation
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
# import serial  # For Arduino communication

# Initialize serial communication with Arduino (commented out)
# arduino = serial.Serial('COM9', 9600)  # Adjust COM port and baud rate as needed
# time.sleep(2)  # Allow time for Arduino to initialize

# Function to send a command to Arduino (commented out)
def send_command_to_arduino(command):
    # arduino.write(f"{command}\n".encode())
    pass


# Function to detect and predict mask
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("mask_detector.h5")

# Initialize the loss function
loss_fn = BinaryCrossentropy()

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

be_ready_start_time = None
countdown_completed = False  # To track whether the countdown is done

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    face_detected = False

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Convert labels to NumPy arrays
        true_label = np.array([1 if mask > withoutMask else 0], dtype=np.float32)
        predicted_label = np.array([mask], dtype=np.float32)

        # Calculate loss
        loss = loss_fn(true_label, predicted_label).numpy() * 100  # Convert to percentage

        # Prepare label and color
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}% | Loss: {:.2f}%".format(label, max(mask, withoutMask) * 100, loss)

        # Draw label and bounding box on the frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        face_detected = True



    if not face_detected:
        text = "WELCOME!"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        send_command_to_arduino("1")  # This call does nothing now
        be_ready_start_time = None

    elif mask > withoutMask:
        if be_ready_start_time is None:  # Start the timer when condition is met
            be_ready_start_time = time.time()
            countdown_completed = False

        elapsed_time = time.time() - be_ready_start_time

        if elapsed_time <= 5:  # Display countdown for the first 5 seconds
            countdown_text = f"Be ready in: {5 - elapsed_time:.2f} sec"
            cv2.putText(frame, countdown_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            countdown_completed = True  # Mark countdown as completed

        if countdown_completed:
            ongoing_text = "ON GOING"
            cv2.putText(frame, ongoing_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2, cv2.LINE_AA)
            send_command_to_arduino("0")  # This call does nothing now

    else:
        text = "Put mask!"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        send_command_to_arduino("1")  # This call does nothing now
        be_ready_start_time = None

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
# arduino.close()  # Commented out since Arduino is not used
