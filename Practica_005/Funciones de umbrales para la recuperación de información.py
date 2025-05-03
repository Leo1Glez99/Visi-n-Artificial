import cv2
import numpy as np
from matplotlib import pyplot as plt

# Cargar la imagen en escala de grises
img = cv2.imread('bookpage.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Umbral binario
_, thresh_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 2. Binario inverso
_, thresh_binary_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# 3. Truncado
_, thresh_trunc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

# 4. To Zero
_, thresh_tozero = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)

# 5. To Zero Inverso
_, thresh_tozero_inv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

# 6. Adaptativo Media
thresh_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

# 7. Adaptativo Gaussiano
thresh_gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

# 8. Otsu
_, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Mostrar resultados en una sola ventana con Matplotlib
titles = ['Original', 'Binary', 'Binary Inv', 'Trunc', 'ToZero', 'ToZero Inv',
          'Adaptativo Media', 'Adaptativo Gauss', 'Otsu']

images = [img, thresh_binary, thresh_binary_inv, thresh_trunc, thresh_tozero,
          thresh_tozero_inv, thresh_mean, thresh_gauss, thresh_otsu]

plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
