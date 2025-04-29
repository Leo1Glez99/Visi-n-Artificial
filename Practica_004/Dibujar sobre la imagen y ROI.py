import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread('watch.jpg')

# Tamaño de la imagen
print("Tamaño de la imagen:", img.shape)  # Debería imprimir (168, 300, 3)

# Dibujar una línea (en la parte superior)
cv2.line(img, (20, 20), (280, 20), (0, 255, 0), 2)

# Dibujar un rectángulo (en el centro)
cv2.rectangle(img, (50, 50), (250, 120), (255, 0, 0), 2)

# Dibujar un círculo (en el centro de la imagen)
cv2.circle(img, (150, 84), 30, (0, 0, 255), -1)

# Escribir texto (abajo de la imagen)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'ROI practica', (50, 160), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

# Definir una región de interés (ROI) dentro de la imagen
roi = img[50:120, 50:250]

# Mostrar la imagen con los dibujos
cv2.imshow('Imagen con Dibujos', img)

# Mostrar solo la ROI recortada
cv2.imshow('ROI - Región de Interés', roi)

cv2.waitKey(0)
cv2.destroyAllWindows()
