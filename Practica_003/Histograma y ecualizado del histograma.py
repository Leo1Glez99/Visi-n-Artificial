import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Cargar la imagen en escala de grises
img = cv2.imread('watch.jpg', cv2.IMREAD_GRAYSCALE)

# 2. Calcular el histograma de la imagen original
hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])

# 3. Ecualizar el histograma de la imagen
img_eq = cv2.equalizeHist(img)

# 4. Calcular el histograma de la imagen ecualizada
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

# 5. Mostrar todo en una sola ventana con subplots
plt.figure(figsize=(10,8))

# Imagen original
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

# Histograma original
plt.subplot(2,2,2)
plt.plot(hist_original, color='black')
plt.title('Histograma Original')
plt.xlim([0,256])

# Imagen ecualizada
plt.subplot(2,2,3)
plt.imshow(img_eq, cmap='gray')
plt.title('Imagen Ecualizada')
plt.axis('off')

# Histograma ecualizado
plt.subplot(2,2,4)
plt.plot(hist_eq, color='black')
plt.title('Histograma Ecualizado')
plt.xlim([0,256])

plt.tight_layout()
plt.show()
