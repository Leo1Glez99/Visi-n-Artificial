import cv2
import numpy as np

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()  # Capturar un cuadro de la cámara
    if not ret:
        print("Error: No se pudo capturar el cuadro.")
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar efecto de negativo
    negative = cv2.bitwise_not(frame)

    # Aplicar desenfoque gaussiano
    blur = cv2.GaussianBlur(frame, (15, 15), 0)

    # Aplicar detección de bordes con Canny
    edges = cv2.Canny(frame, 100, 200)

    # Mostrar las diferentes ventanas con efectos
    cv2.imshow('Original', frame)
    cv2.imshow('Gris', gray)
    cv2.imshow('Negativo', negative)
    cv2.imshow('Desenfoque', blur)
    cv2.imshow('Bordes', edges)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
