import cv2
import numpy as np

# ---------- Cargar imagen ----------
img = cv2.imread("Figuras.jpg")

if img is None:
    print("❌ No se pudo cargar la imagen.")
    exit()

# ---------- Seleccionar ROI ----------
# El usuario selecciona el ROI manualmente con el mouse
print("Selecciona el objeto (ROI) que quieres aislar y presiona ENTER.")
roi = cv2.selectROI("Selecciona ROI", img, showCrosshair=True)
cv2.destroyWindow("Selecciona ROI")

x, y, w, h = roi
recorte = img[y:y+h, x:x+w]

# ---------- Convertir ROI a escala de grises ----------
gray = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)

# ---------- Detección de esquinas ----------
# Usamos el método Shi-Tomasi (bueno para encontrar esquinas)
esquinas = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=100,
    qualityLevel=0.01,
    minDistance=10
)

# Convertimos a entero
esquinas = esquinas.astype(int)

# ---------- Dibujar esquinas ----------
for esquina in esquinas:
    x_esq, y_esq = esquina.ravel()
    cv2.circle(recorte, (x_esq, y_esq), 4, (0, 0, 255), -1)

# ---------- Mostrar resultados ----------
cv2.imshow("ROI con esquinas", recorte)
cv2.waitKey(0)
cv2.destroyAllWindows()
