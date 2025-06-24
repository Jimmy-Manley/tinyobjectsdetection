import cv2
import numpy as np

# Load image (grayscale for simplicity)
img = cv2.imread('low_contrast_image.jpg', cv2.IMREAD_GRAYSCALE)

# 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(img)

# 2. Histogram Equalization
equalized_img = cv2.equalizeHist(img)

# 3. Gamma Correction
gamma = 1.5  # >1 brightens, <1 darkens
gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')

# Show results
cv2.imshow('Original', img)
cv2.imshow('CLAHE', clahe_img)
cv2.imshow('Histogram Equalization', equalized_img)
cv2.imshow('Gamma Correction', gamma_corrected)

cv2.waitKey(0)
cv2.destroyAllWindows()
