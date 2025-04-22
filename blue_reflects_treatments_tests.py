import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, morphology 
import glob
import cv2

import os
# Uses all CPU cores in your computer or Phelma Machines
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["VECLIB_MAXIMUM_THREADS"] = str(os.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())

import gc
gc.collect()

# Destruction des fenêtres ouvertes

cv2.destroyAllWindows()

# Ouverture de l'image

image_right_eye = cv2.imread('Traitement_reflets/right_eye_0534.png')
image_right_eye = cv2.cvtColor(image_right_eye, cv2.COLOR_BGR2RGB)


# Conversion en niveau de gris

image_gray_right_eye = cv2.imread('Traitement_reflets/right_eye_0534.png', cv2.IMREAD_GRAYSCALE)


# Création d'une image HSV
image_hsv = cv2.cvtColor(image_right_eye, cv2.COLOR_RGB2HSV)

lower_blue = np.array([90, 10, 50])  
upper_blue = np.array([140, 255, 255])

lower_blue_2 = np.array([90, 10, 100])  # Adjust these values as needed
upper_blue_2 = np.array([140, 255, 255])

lower_blue_3 = np.array([90, 10, 50])  # Adjust these values as needed
upper_blue_3 = np.array([140, 255, 255])


# Create a mask for blue pixels
blue_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
blue_mask_inverted = blue_mask.copy()

# Invert the mask
for i in range(0, len(blue_mask_inverted)):
    for j in range(0, len(blue_mask_inverted[i])):
        if blue_mask_inverted[i, j] == 0:
            blue_mask_inverted[i, j] = 255
        else:
            blue_mask_inverted[i, j] = 0

blue_detected = cv2.bitwise_and(image_right_eye, image_right_eye, mask=blue_mask)
blue_detected_inverted = cv2.bitwise_and(image_right_eye, image_right_eye, mask=blue_mask_inverted)

#histogramme de l'image en niveau de gris

hist = cv2.calcHist([image_gray_right_eye], [0], None, [256], [0, 256])

# Détection de gradient avec Sobel
sobel_x = cv2.Sobel(image_gray_right_eye, cv2.CV_64F, 1, 0, ksize=3)  # Gradient horizontal
sobel_y = cv2.Sobel(image_gray_right_eye, cv2.CV_64F, 0, 1, ksize=3)  # Gradient vertical

# Calcul de la magnitude du gradient
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Normalisation pour affichage
gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
gradient_magnitude = np.uint8(gradient_magnitude)


#Inversion du gradient final

[ligne, colonne] = gradient_magnitude.shape
for i in range(1, ligne):
    for j in range(1, colonne):
        gradient_magnitude[i, j] = 255 - gradient_magnitude[i, j]

# Création de différentes valeurs à soustraire pour chaque canal

image_right_eye_copy = image_right_eye.copy()
image_right_eye_copy_1 = image_right_eye.copy()
image_right_eye_copy_2 = image_right_eye.copy()

valeur_soustraite = [11,37,126]
valeur_soustraite_1 = [11,37,90]
valeur_soustraite_2 = [11,37,70]
valeur_soustraite_3 = [-11,-37,170]

image_right_eye_copy[blue_mask > 0] = image_right_eye_copy[blue_mask > 0] - valeur_soustraite
image_right_eye_copy_1[blue_mask > 0] = image_right_eye_copy_1[blue_mask > 0] - valeur_soustraite_1
image_right_eye_copy_2[blue_mask > 0] = image_right_eye_copy_2[blue_mask > 0] - valeur_soustraite_2

image_hsv_2 = cv2.cvtColor(image_right_eye_copy, cv2.COLOR_RGB2HSV)
blue_mask_2 = cv2.inRange(image_hsv_2, lower_blue_2, upper_blue_2)
blue_detected_2 = cv2.bitwise_and(image_right_eye_copy, image_right_eye_copy, mask=blue_mask_2)

image_right_eye_copy_copy = image_right_eye_copy.copy()
image_right_eye_copy_copy[blue_mask_2 > 0] = image_right_eye_copy_copy[blue_mask_2 > 0] - valeur_soustraite_3


# Convertir l'image en un tableau NumPy
image_array = image_right_eye.reshape(-1, 3)  # Redimensionner en une liste de pixels (R, G, B)

# Filtrer les pixels non nuls (différents de [0, 0, 0])
non_zero_pixels = image_array[np.any(image_array != [0, 0, 0], axis=1)]

# Calculer la moyenne des pixels non nuls
mean_pixel_value = np.mean(non_zero_pixels, axis=0)

print(f"Moyenne des pixels (R, G, B) sans [0, 0, 0] : {mean_pixel_value}")

# Calcul de la moyenne des pixels du masque bleu

image_array_2 = blue_detected.reshape(-1, 3)  # Redimensionner en une liste de pixels (R, G, B)
non_zero_pixels = image_array_2[np.any(image_array_2 != [0, 0, 0], axis=1)]
mean_pixel_value = np.mean(non_zero_pixels, axis=0)
print(f"Moyenne des pixels (R, G, B) sans [0, 0, 0] : {mean_pixel_value}")



# Affichage des images

fig, axes = plt.subplots(1, 2)

axes[0].imshow(image_right_eye)
axes[0].axis('off')
axes[0].set_title("Image de l'oeuil droit")

axes[1].imshow(image_gray_right_eye, cmap='gray')
axes[1].axis('off')
axes[1].set_title("Image de l'oeuil droit en niveau de gris")


fig_2 = plt.figure() 

plt.plot(hist)
plt.savefig('tmp.png')
nexImg = cv2.imread('tmp.png')

fig3 = plt.figure()
plt.imshow(image_right_eye)
plt.axis('off')
plt.title("Image de base")


fig4, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(sobel_x, cmap='gray')
axes[0].axis('off')
axes[0].set_title("Gradient Sobel X")

axes[1].imshow(sobel_y, cmap='gray')
axes[1].axis('off')
axes[1].set_title("Gradient Sobel Y")

axes[2].imshow(gradient_magnitude, cmap = 'gray')
axes[2].axis('off')
axes[2].set_title("Magnitude du Gradient")

fig5, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].imshow(blue_mask, cmap='gray')
axes[0].axis('off')
axes[0].set_title("Masque des Pixels Bleus")

axes[1].imshow(blue_detected)
axes[1].axis('off')
axes[1].set_title("Pixels Bleus Détectés")

axes[2].imshow(blue_detected_2)
axes[2].axis('off')
axes[2].set_title("Pixels Bleus Détectés avec b = 70")

fig6 = plt.figure()
plt.imshow(image_right_eye_copy)
plt.axis('off')
plt.title("Image avec les pixels moins une constante")

fig7, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_right_eye_copy)
axes[0].axis('off')
axes[0].set_title("b = 126")

axes[1].imshow(image_right_eye_copy_1)
axes[1].axis('off')
axes[1].set_title("b = 90")

axes[2].imshow(image_right_eye_copy_2)
axes[2].axis('off')
axes[2].set_title("b = 70")

fig8 = plt.figure()
plt.imshow(image_right_eye_copy_copy)
plt.axis('off')
plt.title("Image avec les pixels moins une constante, traité deux fois")

plt.show()