import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === Load model and define classes ===
model = load_model("models/waste_model.h5")
classes = ["battery","cardboard","clothes","glass","metal","paper","plastic"]

# Map classes to biodegradable / non-biodegradable
biodegradable_map = {
    "battery": "Non-biodegradable",
    "cardboard": "Biodegradable",
    "clothes": "Biodegradable",
    "glass": "Non-biodegradable",
    "metal": "Non-biodegradable",
    "paper": "Biodegradable",
    "plastic": "Non-biodegradable"
}

# === Variables for smoothing predictions ===
previous_class = None
count = 0
threshold = 3  # number of consecutive frames to confirm prediction

# === Initialize webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    # Bounding box for object placement
    x1, y1 = w // 4, h // 4
    x2, y2 = 3 * w // 4, 3 * h // 4
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Crop ROI inside bounding box
    roi = frame[y1:y2, x1:x2]
    img = cv2.resize(roi, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    preds = model.predict(img, verbose=0)
    class_id = np.argmax(preds[0])
    label = classes[class_id]
    confidence = preds[0][class_id]*100

    # Smoothing
    if label == previous_class:
        count += 1
    else:
        count = 1
        previous_class = label

    if count >= threshold:
        final_label = label
        final_conf = confidence
    else:
        final_label = "Waiting..."
        final_conf = 0

    # Display overlay text
    if final_label != "Waiting...":
        bio_label = biodegradable_map[final_label]
        display_text = f"{final_label} ({bio_label})"
    else:
        display_text = final_label

    # Draw filled rectangle for text background
    cv2.rectangle(frame, (0,0), (w, 40), (0,0,0), -1)  # black rectangle
    cv2.putText(frame, display_text, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Waste Segregation", frame)

    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()