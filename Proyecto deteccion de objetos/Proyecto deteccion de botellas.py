import cv2
from ultralytics import YOLO
import serial
import time

# Configurar puerto Serial
esp32 = serial.Serial('COM4', 115200)  # Ajusta según tu puerto
time.sleep(2)  # Espera a que el Serial esté listo

model = YOLO('yolov8n.pt')  # Modelo pequeño

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        result = results[0]

        botella_detectada = False

        for box, cls, score in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_name = model.names[int(cls)]

            if class_name == 'bottle' and float(score) > 0.5:
                botella_detectada = True
                x1, y1, x2, y2 = box.int().tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Bottle {score:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if botella_detectada:
            esp32.write(b'1')
        else:
            esp32.write(b'0')

        cv2.imshow("Detección Botella", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    esp32.close()
    cv2.destroyAllWindows()
