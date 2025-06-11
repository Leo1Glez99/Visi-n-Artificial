import cv2
import numpy as np

# Captura de video
cap = cv2.VideoCapture(0)

# Subtractor de fondo MOG2
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener la máscara del fondo (blanco = movimiento)
    fgmask = fgbg.apply(frame)

    # Aplicar filtro morfológico para limpiar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Convertir la máscara binaria en máscara de 3 canales para usarla con el frame en color
    mask_3ch = cv2.merge([fgmask, fgmask, fgmask])

    # Aplicar la máscara al frame original (bitwise AND)
    foreground = cv2.bitwise_and(frame, mask_3ch)

    # Mostrar resultados
    cv2.imshow("Original", frame)
    cv2.imshow("Fondo Eliminado (color)", foreground)

    # Salir con 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
