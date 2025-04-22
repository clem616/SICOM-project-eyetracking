# # -*- coding: utf-8 -*-
# """
# Created on Wed Mar 19 08:29:44 2025

# @author: marti
# """



# from scipy.ndimage import affine_transform
# from sklearn.metrics import confusion_matrix
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.pyplot as plt

# import numpy as np
# import os
# import gc
# import random
# import time
# from skimage.filters import threshold_otsu
# from skimage.morphology import diamond,disk,square, remove_small_objects, remove_small_holes,opening,closing,dilation,erosion
# import glob
# from PIL import Image
# import cv2

# from matplotlib import pyplot as plt

# import pandas as pd
# from tqdm import tqdm

# #image_path = '/home/clem/Documents/my-eye-tracking/neural_network/bas_droit/left_eye_0111.png'
# #image_path = '/home/clem/Documents/my-eye-tracking/neural_network/haut_droit/left_eye_0001.png'

# def average_image_size(dataset_path):
#  total_images = 0
#  total_width = 0
#  total_height = 0
#  for root, dirs, files in os.walk(dataset_path):
#      for file in files:
#          if file.endswith(('.png', '.jpg', '.jpeg')):
#              image_path = os.path.join(root, file)
#              image = Image.open(image_path)
#              width, height = image.size
#              total_width += width
#              total_height += height
#              total_images += 1
#  avg_width = round(total_width / total_images)
#  avg_height = round(total_height / total_images)
#  return avg_width, avg_height

# dataset_path = "/home/clem/Documents/my-eye-tracking/dataset_clement_5/oeil_gauche"
# target_size = average_image_size(dataset_path)

# def load_image(image_path, target_size = target_size):
#    image = Image.open(image_path)
#    image = image.convert('RGB')  # Convertir l'image en mode RGB
#    image = image.resize(target_size)  # Redimensionner l'image
#    image_array = np.array(image)
#    #print(f"Image: {image_path}, Taille: {image_array.shape}")
#    return image_array



# def barycentre(image):
#    # Convertir le chemin de l'image en tableau NumPy
#    #image = load_image(image_path)



#    # Conversion en niveaux de gris
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#    # Calcul du gradient de l'image en niveaux de gris
#    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

#    # Calcul de la norme du gradient
#    grad_norm = np.sqrt(grad_x**2 + grad_y**2)

#    # Normalisation de la norme du gradient pour qu'elle soit entre 0 et 255
#    grad_norm = cv2.normalize(grad_norm, None, 0, 255, cv2.NORM_MINMAX)

#    # Conversion en type de données 8 bits
#    grad_norm = grad_norm.astype(np.uint8)

#    # Création du masque binaire en utilisant un seuil
#    _, mask = cv2.threshold(grad_norm, 50, 255, cv2.THRESH_BINARY)

#    #Affichage du masque
#    # plt.figure(figsize=(5, 5))
#    # plt.imshow(mask, cmap='gray')
#    # plt.title('Masque de la norme du gradient')
#    # plt.axis('off')
#    # plt.show()





#    ########################## Vecteur direction de l'oeil

#    equ = cv2.equalizeHist(gray)

#    # # # On affiche les points où le pixel vaut 0
#    mask_zero = cv2.inRange(equ, 0, 0)


#    # Récupère les indices (y, x) des pixels où mask_zero vaut 255
#    white_coords = np.column_stack(np.where(mask_zero == 255))
#    print(white_coords)

#    # Calcule le barycentre des pixels blancs
#    barycenter = white_coords.mean(axis=0)
  
#    return barycenter


# def eye_opening_size(image):
#     # Convertir l'image en niveaux de gris
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Appliquer un flou pour réduire le bruit
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Seuillage adaptatif pour gérer les variations d’éclairage
#     thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                     cv2.THRESH_BINARY_INV, 11, 2)
    
#     # Trouver les contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     eye_opening_area = 0
#     for contour in contours:
#         if len(contour) > 5:
#             # Trouver le cercle englobant
#             (x, y), radius = cv2.minEnclosingCircle(contour)
#             area = np.pi * (radius ** 2)  # Aire du cercle
            
#             if area > eye_opening_area:
#                 eye_opening_area = area

#     # Normalisation par rapport à la taille de l’image pour éviter les effets de zoom
#     eye_width = image.shape[1]  # Largeur de l'image de l'œil
#     if eye_width > 0:
#         eye_opening_area /= (eye_width ** 2)

#     return eye_opening_area


# def visualize_eye_detection(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY_INV, 11, 2)
    
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for contour in contours:
#         if len(contour) > 5:
#             # Trouver l'ellipse ou cercle
#             ellipse = cv2.fitEllipse(contour)
#             cv2.ellipse(image, ellipse, (0, 255, 0), 2)  # Dessiner l'ellipse
#             (center, (major_axis, minor_axis), angle) = ellipse
#             ellipse_params = (center, major_axis, minor_axis, angle)
    
#     # Affichage de l'image avec l'ellipse tracée
#     # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     # plt.title("Détection de l'œil")
#     # plt.axis('off')
#     # plt.show()
#     return ellipse_params


# # Fonction qui met les données dans un ficher .csv  (base de données)
# def process_dataset(dataset_path, output_file):
#       data = []
      
#       # Parcours des sous-dossiers (directions du regard)
#       for direction in os.listdir(dataset_path):
#           direction_path = os.path.join(dataset_path, direction)
#           if not os.path.isdir(direction_path):
#               continue
          
#           image_files = [f for f in os.listdir(direction_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
          
#           for image_file in tqdm(image_files, desc=f"Processing {direction}"):
#               image_path = os.path.join(direction_path, image_file)
#               image = load_image(image_path)
#               barycenter = barycentre(image)
#               data.append([direction, float(barycenter[0]), float(barycenter[1])])
      
#       # Sauvegarde en CSV avec séparateur correct
#       df = pd.DataFrame(data, columns=['direction', 'barycenter_x', 'barycenter_y'])
#       df.to_csv(output_file, index=False, sep=';', encoding='utf-8')
#       print(f"Barycentres saved in {output_file}")

# def process_dataset_ellipse(dataset_path, output_file):
#     data = []
    
#     # Parcours des sous-dossiers (directions du regard)
#     for direction in os.listdir(dataset_path):
#         direction_path = os.path.join(dataset_path, direction)
#         if not os.path.isdir(direction_path):
#             continue
        
#         image_files = [f for f in os.listdir(direction_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
#         for image_file in tqdm(image_files, desc=f"Processing {direction}"):
#             image_path = os.path.join(direction_path, image_file)
#             image = load_image(image_path)  # Fonction pour charger l'image
#             barycenter = barycentre(image)  # Fonction pour calculer le barycentre
            
#             # Extraction des paramètres de l'ellipse
#             ellipse_params = visualize_eye_detection(image)
#             if ellipse_params:
#                 center, major_axis, minor_axis, angle = ellipse_params
#                 center_x, center_y = center
#                 data.append([
#                     direction, 
#                     round(float(barycenter[0]),1), round(float(barycenter[1]),1), 
#                     round(float(center_x),1), round(float(center_y),1), 
#                     round(float(major_axis),1), round(float(minor_axis),1), round(float(angle),1)
#                 ])
    
#     # Sauvegarde en CSV avec séparateur correct
#     df = pd.DataFrame(data, columns=[
#         'direction', 'barycenter_x', 'barycenter_y', 
#         'center_x', 'center_y', 'major_axis', 'minor_axis', 'angle'
#     ])
#     df.to_csv(output_file, index=False, sep=';', encoding='utf-8')
#     print(f"Données sauvegardées dans {output_file}")
    
    
# def process_dataset_with_opening(dataset_path, output_file):
#     data = []
    
#     # Parcours des sous-dossiers (directions du regard)
#     for direction in os.listdir(dataset_path):
#         direction_path = os.path.join(dataset_path, direction)
#         if not os.path.isdir(direction_path):
#             continue
        
#         image_files = [f for f in os.listdir(direction_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
#         for image_file in tqdm(image_files, desc=f"Processing {direction}"):
#             image_path = os.path.join(direction_path, image_file)
#             image = load_image(image_path)
#             barycenter = barycentre(image)
#             eye_size = eye_opening_size(image)
#             data.append([direction, float(barycenter[0]), float(barycenter[1]), float(eye_size)])
    
#     # Sauvegarde en CSV avec séparateur correct
#     df = pd.DataFrame(data, columns=['direction', 'barycenter_x', 'barycenter_y', 'eye_size'])
#     df.to_csv(output_file, index=False, sep=';', encoding='utf-8')
#     print(f"Barycentres and eye sizes saved in {output_file}")
 
# output_file = "features_oeil_gauche.csv"
# # output_file = "barycentres_Martin_avgsize.csv"
# process_dataset_ellipse(dataset_path, output_file)






# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 08:29:44 2025

@author: marti
"""



from scipy.ndimage import affine_transform
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

import numpy as np
import os
import gc
import random
import time
from skimage.filters import threshold_otsu
from skimage.morphology import diamond,disk,square, remove_small_objects, remove_small_holes,opening,closing,dilation,erosion
import glob
from PIL import Image
import cv2

from matplotlib import pyplot as plt

import pandas as pd
from tqdm import tqdm

#image_path = '/home/clem/Documents/my-eye-tracking/neural_network/bas_droit/left_eye_0111.png'
#image_path = '/home/clem/Documents/my-eye-tracking/neural_network/haut_droit/left_eye_0001.png'

def average_image_size(dataset_path):
 total_images = 0
 total_width = 0
 total_height = 0
 for root, dirs, files in os.walk(dataset_path):
     for file in files:
         if file.endswith(('.png', '.jpg', '.jpeg')):
             image_path = os.path.join(root, file)
             image = Image.open(image_path)
             width, height = image.size
             total_width += width
             total_height += height
             total_images += 1
 avg_width = round(total_width / total_images)
 avg_height = round(total_height / total_images)
 return avg_width, avg_height

dataset_path = "/home/clem/Documents/my-eye-tracking/dataset_clement_6/left_eye"
target_size = average_image_size(dataset_path)

def load_image(image_path, target_size = target_size):
   image = Image.open(image_path)
   image = image.convert('RGB')  # Convertir l'image en mode RGB
   image = image.resize(target_size)  # Redimensionner l'image
   image_array = np.array(image)
   #print(f"Image: {image_path}, Taille: {image_array.shape}")
   return image_array



def barycentre(image):
   # Convertir le chemin de l'image en tableau NumPy
   #image = load_image(image_path)



   # Conversion en niveaux de gris
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   # Calcul du gradient de l'image en niveaux de gris
   grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
   grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

   # Calcul de la norme du gradient
   grad_norm = np.sqrt(grad_x**2 + grad_y**2)

   # Normalisation de la norme du gradient pour qu'elle soit entre 0 et 255
   grad_norm = cv2.normalize(grad_norm, None, 0, 255, cv2.NORM_MINMAX)

   # Conversion en type de données 8 bits
   grad_norm = grad_norm.astype(np.uint8)

   # Création du masque binaire en utilisant un seuil
   _, mask = cv2.threshold(grad_norm, 50, 255, cv2.THRESH_BINARY)

   #Affichage du masque
   # plt.figure(figsize=(5, 5))
   # plt.imshow(mask, cmap='gray')
   # plt.title('Masque de la norme du gradient')
   # plt.axis('off')
   # plt.show()





   ########################## Vecteur direction de l'oeil

   equ = cv2.equalizeHist(gray)

   # # # On affiche les points où le pixel vaut 0
   mask_zero = cv2.inRange(equ, 0, 0)


   # Récupère les indices (y, x) des pixels où mask_zero vaut 255
   white_coords = np.column_stack(np.where(mask_zero == 255))
   print(white_coords)

   # Calcule le barycentre des pixels blancs
   barycenter = white_coords.mean(axis=0)
  
   return barycenter


def eye_opening_size(image):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Seuillage adaptatif pour gérer les variations d’éclairage
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Trouver les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    eye_opening_area = 0
    for contour in contours:
        if len(contour) > 5:
            # Trouver le cercle englobant
            (x, y), radius = cv2.minEnclosingCircle(contour)
            area = np.pi * (radius ** 2)  # Aire du cercle
            
            if area > eye_opening_area:
                eye_opening_area = area

    # Normalisation par rapport à la taille de l’image pour éviter les effets de zoom
    eye_width = image.shape[1]  # Largeur de l'image de l'œil
    if eye_width > 0:
        eye_opening_area /= (eye_width ** 2)

    return eye_opening_area


def visualize_eye_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) > 5:
            # Trouver l'ellipse ou cercle
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)  # Dessiner l'ellipse
            (center, (major_axis, minor_axis), angle) = ellipse
            ellipse_params = (center, major_axis, minor_axis, angle)
    
    # Affichage de l'image avec l'ellipse tracée
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title("Détection de l'œil")
    # plt.axis('off')
    # plt.show()
    return ellipse_params


# Fonction qui met les données dans un ficher .csv  (base de données)
def process_dataset(dataset_path, output_file):
      data = []
      
      # Parcours des sous-dossiers (directions du regard)
      for direction in os.listdir(dataset_path):
          direction_path = os.path.join(dataset_path, direction)
          if not os.path.isdir(direction_path):
              continue
          
          image_files = [f for f in os.listdir(direction_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
          
          for image_file in tqdm(image_files, desc=f"Processing {direction}"):
              image_path = os.path.join(direction_path, image_file)
              image = load_image(image_path)
              barycenter = barycentre(image)
              data.append([direction, float(barycenter[0]), float(barycenter[1])])
      
      # Sauvegarde en CSV avec séparateur correct
      df = pd.DataFrame(data, columns=['direction', 'barycenter_x', 'barycenter_y'])
      df.to_csv(output_file, index=False, sep=';', encoding='utf-8')
      print(f"Barycentres saved in {output_file}")

def process_dataset_ellipse(dataset_path, output_file):
    data = []
    
    # Parcours des sous-dossiers (directions du regard)
    for direction in os.listdir(dataset_path):
        direction_path = os.path.join(dataset_path, direction)
        if not os.path.isdir(direction_path):
            continue
        
        image_files = [f for f in os.listdir(direction_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in tqdm(image_files, desc=f"Processing {direction}"):
            image_path = os.path.join(direction_path, image_file)
            image = load_image(image_path)  # Fonction pour charger l'image
            barycenter = barycentre(image)  # Fonction pour calculer le barycentre
            
            # Extraction des paramètres de l'ellipse
            ellipse_params = visualize_eye_detection(image)
            if ellipse_params:
                center, major_axis, minor_axis, angle = ellipse_params
                center_x, center_y = center
                data.append([
                    direction, 
                    round(float(barycenter[0]),1), round(float(barycenter[1]),1), 
                    round(float(center_x),1), round(float(center_y),1), 
                    round(float(major_axis),1), round(float(minor_axis),1), round(float(angle),1)
                ])
    
    # Sauvegarde en CSV avec séparateur correct
    df = pd.DataFrame(data, columns=[
        'direction', 'barycenter_x', 'barycenter_y', 
        'center_x', 'center_y', 'major_axis', 'minor_axis', 'angle'
    ])
    df.to_csv(output_file, index=False, sep=';', encoding='utf-8')
    print(f"Données sauvegardées dans {output_file}")
    
    
def process_dataset_with_opening(dataset_path, output_file):
    data = []
    
    # Parcours des sous-dossiers (directions du regard)
    for direction in os.listdir(dataset_path):
        direction_path = os.path.join(dataset_path, direction)
        if not os.path.isdir(direction_path):
            continue
        
        image_files = [f for f in os.listdir(direction_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in tqdm(image_files, desc=f"Processing {direction}"):
            image_path = os.path.join(direction_path, image_file)
            image = load_image(image_path)
            barycenter = barycentre(image)
            eye_size = eye_opening_size(image)
            data.append([direction, float(barycenter[0]), float(barycenter[1]), float(eye_size)])
    
    # Sauvegarde en CSV avec séparateur correct
    df = pd.DataFrame(data, columns=['direction', 'barycenter_x', 'barycenter_y', 'eye_size'])
    df.to_csv(output_file, index=False, sep=';', encoding='utf-8')
    print(f"Barycentres and eye sizes saved in {output_file}")
 
output_file = "features_left_eye.csv"
# output_file = "barycentres_Martin_avgsize.csv"
process_dataset_ellipse(dataset_path, output_file)

