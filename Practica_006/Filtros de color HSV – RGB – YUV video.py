import cv2
import numpy as np

# Captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. Filtro rojo (requiere 2 rangos por cómo está distribuido el rojo en HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # 2. Filtro verde
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 3. Filtro azul
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Aplicar las máscaras
    red_result = cv2.bitwise_and(frame, frame, mask=mask_red)
    green_result = cv2.bitwise_and(frame, frame, mask=mask_green)
    blue_result = cv2.bitwise_and(frame, frame, mask=mask_blue)

    # Mostrar resultados
    cv2.imshow('Original', frame)
    cv2.imshow('Rojo', red_result)
    cv2.imshow('Verde', green_result)
    cv2.imshow('Azul', blue_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
