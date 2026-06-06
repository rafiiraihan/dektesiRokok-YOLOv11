import cv2
import time
import os
from ultralytics import YOLO

# =========================
# KONFIGURASI
# =========================
MODEL_PATH = "deteksiRokokYolov11.pt"      # ganti jika nama model berbeda
CAMERA_INDEX = 0               # 0 = webcam utama
DETECTION_TIME = 10            # deteksi kontinu (detik)
SAVE_DIR = "screenshots"       # folder screenshot

# =========================
# PERSIAPAN
# =========================
os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
camera = cv2.VideoCapture(CAMERA_INDEX)

detected_start_time = None
screenshot_taken = False
prev_time = time.time()

# =========================
# LOOP UTAMA
# =========================
while True:
    ret, frame = camera.read()
    if not ret:
        print("[ERROR] Kamera tidak dapat dibuka.")
        break

    # YOLO inference
    results = model(frame, verbose=False)

    # Cek apakah ada objek terdeteksi
    detected = len(results[0].boxes) > 0

    current_time = time.time()

    # Hitung FPS
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    if detected:
        if detected_start_time is None:
            detected_start_time = current_time

        elapsed_time = current_time - detected_start_time

        # Screenshot jika terdeteksi >= 10 detik
        if elapsed_time >= DETECTION_TIME and not screenshot_taken:
            filename = f"rokok_detected_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)

            print(f"[INFO] Screenshot disimpan: {filepath}")
            screenshot_taken = True
    else:
        # Reset jika objek hilang
        detected_start_time = None
        screenshot_taken = False

    # Visualisasi hasil deteksi
    plotted_frame = results[0].plot()

    # Tampilkan FPS
    cv2.putText(
        plotted_frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Tampilkan status deteksi
    status_text = "Rokok Terdeteksi" if detected else "Tidak Ada Rokok"
    status_color = (0, 0, 255) if detected else (0, 255, 0)

    cv2.putText(
        plotted_frame,
        status_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )

    cv2.imshow("Deteksi Rokok YOLOv11", plotted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
camera.release()
cv2.destroyAllWindows()
