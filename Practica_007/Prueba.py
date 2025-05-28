import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Usa 1 si tu cámara es externa: cv2.VideoCapture(1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango para color rojo
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Operaciones morfológicas
    tophat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernel)

    # Máscara corregida
    cleaned_mask = cv2.subtract(mask, blackhat)
    cleaned_mask = cv2.add(cleaned_mask, tophat)

    # Resultado usando máscara corregida
    result = cv2.bitwise_and(frame, frame, mask=cleaned_mask)

    # Mostrar ventanas
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Tophat', tophat)
    cv2.imshow('Blackhat', blackhat)
    cv2.imshow('Cleaned Mask', cleaned_mask)
    cv2.imshow('Result', result)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
