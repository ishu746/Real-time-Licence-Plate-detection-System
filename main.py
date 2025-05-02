from ultralytics import YOLO
import cv2
import re
from collections import defaultdict
import time
from paddleocr import PaddleOCR
import csv
import os
import threading

# --- Function to extract plate text ---
def readPlateText(plateImage, ocr):
    result = ocr.ocr(plateImage, cls=False)

    if not result[0]:
        return None, 0

    maxArea = 0
    largestText = None
    largestTextConfidence = 0

    for idx in range(len(result)):
        res = result[idx]
        for bbox, detect in res:
            x1, y1 = bbox[0]
            x2, y2 = bbox[-2]
            area = (x2 - x1) * (y2 - y1)

            if area > maxArea:
                maxArea = area
                largestText = detect[0].upper()
                largestTextConfidence = detect[1]

    cleanText = re.sub(r'[^a-zA-Z0-9]', '', largestText)
    return cleanText, largestTextConfidence


def process_frame(frame, plateModel, ocr, plateTracker, seenPlates, csv_filename):
    detections = plateModel.track(frame, conf=0.7, persist=True, verbose=False)[0]
    currIDs = set()

    for detection in detections.boxes.data.tolist():
        if len(detection) != 7:
            continue
        x1, y1, x2, y2, id, score, _ = detection
        currIDs.add(id)

        platCrop = frame[int(y1):int(y2), int(x1):int(x2)].copy()
        plateText, confidence = readPlateText(platCrop, ocr)

        confThreshold = 0.05
        if plateText:
            currConf = plateTracker[id]['confidence']
            currText = plateTracker[id]['text']

            if confidence > currConf or (abs(confidence - currConf) <= confThreshold and len(plateText) > len(currText)):
                plateTracker[id]['text'] = plateText
                plateTracker[id]['confidence'] = confidence
                plateTracker[id]['counter'] = 0

            plate_key = f"{int(id)}_{plateText}"
            if plate_key not in seenPlates:
                seenPlates.add(plate_key)
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        time.strftime('%Y-%m-%d %H:%M:%S'),
                        int(id),
                        plateText,
                        round(confidence, 2)
                    ])
                    file.flush()

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return currIDs


if __name__ == '__main__':
    ocr = PaddleOCR(lang='en')
    plateModel = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture('media/test.mp4')
    # cap = cv2.VideoCapture(0)             # Uncomment for webcam



    fps = cap.get(cv2.CAP_PROP_FPS)
    frameTime = 1 / fps if fps > 0 else 1 / 30

    w, h = 1280, 720
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', w, h)

    plateTracker = defaultdict(lambda: {'text': None, 'confidence': 0, 'counter': 0})

    csv_filename = 'detected_plates.csv'
    seenPlates = set()

    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['FrameTime', 'PlateID', 'Text', 'Confidence'])

    ret = True
    while ret:
        startTime = time.time()
        ret, frame = cap.read()

        if not ret:
            break

        thread = threading.Thread(target=process_frame, args=(frame, plateModel, ocr, plateTracker, seenPlates, csv_filename))
        thread.start()
        thread.join()

        # Timeout tracking
        currIDs = set(plateTracker.keys())
        idToRemove = []
        for id in plateTracker:
            if id not in currIDs:
                plateTracker[id]['counter'] += 1
                if plateTracker[id]['counter'] > 5:
                    idToRemove.append(id)

        for id in idToRemove:
            del plateTracker[id]

        for plateID, plateData in plateTracker.items():
            if plateData['text']:
                label = f"ID: {plateID} | {plateData['text']} ({plateData['confidence']:.2f})"
                cv2.putText(frame, label, (20, 30 + 30 * int(plateID)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)

        elapsed = time.time() - startTime
        delay = max(int((frameTime - elapsed) * 1000), 1)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Real-time detections saved to: {csv_filename}")
