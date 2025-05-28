import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Kernel morfológico
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango para rojo
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Rango para verde
    lower_green = np.array([36, 50, 70])
    upper_green = np.array([89, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Rango para azul
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Máscara combinada
    combined_mask = cv2.bitwise_or(mask_red, mask_green)
    combined_mask = cv2.bitwise_or(combined_mask, mask_blue)

    # 🔍 Aplicamos TopHat y BlackHat a la máscara para limpiar detalles
    tophat = cv2.morphologyEx(combined_mask, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(combined_mask, cv2.MORPH_BLACKHAT, kernel)

    # 🧼 Limpiamos la máscara combinando resultados
    cleaned_mask = cv2.subtract(combined_mask, blackhat)
    cleaned_mask = cv2.add(cleaned_mask, tophat)

    # F+ (detección positiva)
    F_pos = cv2.bitwise_and(frame, frame, mask=cleaned_mask)

    # F- (resto de la imagen)
    mask_inv = cv2.bitwise_not(cleaned_mask)
    F_neg = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Mostrar solo F+ y F-
    cv2.imshow('F+', F_pos)
    cv2.imshow('F-', F_neg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
