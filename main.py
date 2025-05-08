import cv2
import json
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db

load_dotenv()
firebase_db_url = os.getenv("FIREBASE_DB_URL")

# ------------------- FIREBASE SETUP -------------------

cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': firebase_db_url
})
firebase_ref = db.reference("vehicle_data")

# ------------------- VEHICLE DETECTION -------------------

fgbg = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture("traffic.mp4")

lane_boundaries = [(0, 300), (300, 600), (600, 900)]  # Adjust these values as needed
vehicle_data = []

CONGESTION_THRESHOLD = 10  # Threshold to mark a lane as congested

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (1100, 900))
        fgmask = fgbg.apply(resized_frame)
        _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lane_vehicle_counts = [0] * len(lane_boundaries)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)

                for i, (lane_start, lane_end) in enumerate(lane_boundaries):
                    if lane_start <= x <= lane_end:
                        lane_vehicle_counts[i] += 1
                        break

                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 🚦 Detect congested lanes
        prioritized_lanes = []
        for i, count in enumerate(lane_vehicle_counts):
            if count >= CONGESTION_THRESHOLD:
                prioritized_lanes.append(i + 1)

        # 💾 Store data locally
        frame_data = {
            "frame": len(vehicle_data) + 1,
            "lane_counts": lane_vehicle_counts,
            "prioritized_lanes": prioritized_lanes
        }
        vehicle_data.append(frame_data)

        # 🔥 Push to Firebase
        firebase_ref.push(frame_data)

        # 🖼️ Draw lanes and info
        for i, (lane_start, _) in enumerate(lane_boundaries):
            color = (0, 0, 255) if (i + 1) in prioritized_lanes else (255, 0, 0)
            cv2.line(resized_frame, (lane_start, 0), (lane_start, resized_frame.shape[0]), color, 2)
            cv2.putText(resized_frame, f"Lane {i+1}: {lane_vehicle_counts[i]}",
                        (lane_start + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Detected Vehicles", resized_frame)
        cv2.imshow("Mask", fgmask)

        print("Vehicles per lane:", lane_vehicle_counts)
        if prioritized_lanes:
            print("⚠️ Prioritized lanes due to congestion:", prioritized_lanes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    with open("vehicle_data.json", "w") as json_file:
        json.dump(vehicle_data, json_file, indent=4)
    print("✅ Vehicle data saved to 'vehicle_data.json'")
