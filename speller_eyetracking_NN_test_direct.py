# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import seaborn as sns

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping



from collections import Counter

from scipy.ndimage import affine_transform
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

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

import cv2
import mediapipe as mp
import os
import time
import numpy as np
import glob
import random
from skimage.filters import threshold_otsu
from skimage.morphology import diamond,disk,square, remove_small_objects, remove_small_holes,opening,closing,dilation,erosion
from PIL import Image
from collections import Counter



def feuille_numerique(coin_choisit_1, coin_choisit_2):
    if coin_choisit_1 == 0:
        if coin_choisit_2 == 0:   caractere = "a"
        elif coin_choisit_2 == 1: caractere = "b"
        elif coin_choisit_2 == 2: caractere = "c"
        elif coin_choisit_2 == 3: caractere = "d"
        elif coin_choisit_2 == 4: caractere = "f"
        elif coin_choisit_2 == 5: caractere = " "
    elif coin_choisit_1 == 1:
        if coin_choisit_2 == 0:   caractere = "g"
        elif coin_choisit_2 == 1: caractere = "h"
        elif coin_choisit_2 == 2: caractere = "i"
        elif coin_choisit_2 == 3: caractere = "j"
        elif coin_choisit_2 == 4: caractere = "k"
        elif coin_choisit_2 == 5: caractere = " suppresseur"
    elif coin_choisit_1 == 2:
        if coin_choisit_2 ==0:    caractere = "l"
        elif coin_choisit_2 == 1: caractere = "m"
        elif coin_choisit_2 == 2: caractere = "n"
        elif coin_choisit_2 == 3: caractere = "o"
        elif coin_choisit_2 == 4: caractere = "p"
        elif coin_choisit_2 == 5: caractere = "suppr_mot"
    elif coin_choisit_1 ==3:
        if coin_choisit_2 == 0:   caractere = "q"
        elif coin_choisit_2 == 1: caractere = "r"
        elif coin_choisit_2 == 2: caractere = "s"
        elif coin_choisit_2 == 3: caractere = "t"
        elif coin_choisit_2 == 4: caractere = "u"
        elif coin_choisit_2 == 5: caractere = "entrée"
    elif coin_choisit_1 == 5:
        if coin_choisit_2 ==0:    caractere = "v"
        elif coin_choisit_2 == 1: caractere = "w"
        elif coin_choisit_2 == 2: caractere = "x"
        elif coin_choisit_2 == 3: caractere = "y"
        elif coin_choisit_2 == 4: caractere = "z"
        elif coin_choisit_2 == 5: caractere = "e"
    # else:
    else:
        return "0"
    #     caractere = verso_feuille(coin_choisit_1, coin_choisit_2)
    return caractere






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

#dataset_path = "/home/clem/Documents/my-eye-tracking/dataset_clem_6/left_eye"

def load_image(image_path, target_size):
   image = Image.open(image_path)
   image = image.convert('RGB')  # Convertir l'image en mode RGB
   image = image.resize(target_size)  # Redimensionner l'image
   image_array = np.array(image)
   return image_array



def barycentre(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_norm = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = cv2.normalize(grad_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(grad_norm, 50, 255, cv2.THRESH_BINARY)
    equ = cv2.equalizeHist(gray)
    mask_zero = cv2.inRange(equ, 0, 0)
    white_coords = np.column_stack(np.where(mask_zero == 255))
    barycenter = white_coords.mean(axis=0, dtype=np.float64)  # [y, x]
    return barycenter







# -------------------------------------------------------------------
# ------------------------- PHASE DE CALIBRATION ---------------------
# -------------------------------------------------------------------###########################################

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Liste des directions à capturer
directions = [
    "milieu_haut",
    "haut_gauche",
    "bas_gauche",
    "milieu_bas",
    "bas_droit",
    "haut_droit"
]



root_output_dir = "/home/clem/Documents/my-eye-tracking/final_round/dataset"

# Indices des landmarks des yeux
left_eye_indices = [33, 133, 157, 158, 159, 160, 161, 144, 145, 153, 154, 155]

right_eye_indices = [362, 398, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385]


##################################################SET TIME#######################################################TIME TIME###############
duration = 10  # secondes
fps = 30
interval = 1 / fps

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
    exit(1)

L = []  # Liste des barycentres moyens pour chaque direction
O2 = [] # Liste des ouvertures verticales (toutes directions confondues)

for direction in directions:
    # Au lieu de base_output_dir = os.path.join(root_output_dir, direction),
    # on reste simplement sur le root (dataset).
    base_output_dir = root_output_dir
    
    # On définit un répertoire par œil, *puis* la direction en sous-dossier
    eye_dirs = {
        "left_eye":  os.path.join(base_output_dir, "left_eye",  direction),
        "right_eye": os.path.join(base_output_dir, "right_eye", direction)
    }

    # Création des dossiers
    for dir_path in eye_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Fonction pour récupérer l’index de départ dans chaque dossier
    def get_last_index(directory):
        existing_files = os.listdir(directory)
        images = [f for f in existing_files if f.endswith(".png")]
        # On retourne 0 si aucun fichier n’existe
        if not images:
            return 0
        return max(int(f.split("_")[-1].split(".")[0]) for f in images)

    # Compteurs séparés par œil
    image_counters = {
        "left_eye":  get_last_index(eye_dirs["left_eye"]),
        "right_eye": get_last_index(eye_dirs["right_eye"]),
    }
    # Pour l'affichage final du nombre d'images par direction
    # (on prend la plus grande valeur actuelle pour initialiser frame_count)
    frame_count = max(image_counters.values())

    print(f"Capture {direction}, prépare-toi...")
    time.sleep(3)
    print(f"Enregistrement des images pour {direction}...")

    start_time = time.time()
    O = []  # Ouvertures verticales pour cette direction

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la capture vidéo.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # On boucle sur l'œil droit puis l'œil gauche
                for eye_indices, eye_name in zip([right_eye_indices, left_eye_indices],
                                                 ["right_eye", "left_eye"]):
                    try:
                        x_coords = [int(face_landmarks.landmark[idx].x * frame.shape[1]) for idx in eye_indices]
                        y_coords = [int(face_landmarks.landmark[idx].y * frame.shape[0]) for idx in eye_indices]

                        x_min = max(0, min(x_coords) - 10)
                        x_max = min(frame.shape[1], max(x_coords) + 10)
                        y_min = max(0, min(y_coords) - 10)
                        y_max = min(frame.shape[0], max(y_coords) + 10)

                        ouverture = abs(y_max - y_min)
                        O.append(ouverture)

                        eye_region = frame[y_min:y_max, x_min:x_max]
                        if eye_region.size > 0:
                            frame_count += 1
                            image_counters[eye_name] += 1

                            filename = os.path.join(
                                eye_dirs[eye_name],
                                f"{eye_name}_{image_counters[eye_name]:04d}.png"
                            )
                            cv2.imwrite(filename, eye_region)
                            cv2.imshow(eye_name, eye_region)
                    except Exception as e:
                        print(f"Erreur dans l'extraction de {eye_name}: {e}")

        cv2.imshow('Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
        time.sleep(interval)

    # Après la boucle while, on stocke la liste des ouvertures pour cette direction
    O2.append(O)
    print(f"Capture terminée pour {direction} - {frame_count} images enregistrées.")

    # Calcul de la taille moyenne des images :
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
        # On évite la division par 0 si pas d'image :
        if total_images == 0:
            return (0, 0)
        avg_width = round(total_width / total_images)
        avg_height = round(total_height / total_images)
        return avg_width, avg_height

    # Ici on reprend la même logique : on calcule la taille moyenne
    # dans tout le dossier base_output_dir ( = dataset ), si on veut.
    target_size = average_image_size(base_output_dir)

    # Calcul du barycentre moyen pour cette direction
    barycentre_moy = np.array([0.0, 0.0])
    t = 0
    # Parcours récursif de tout le dossier "dataset", on peut ajuster si besoin
    for image_file in glob.glob(os.path.join(base_output_dir, '**', '*.png'), recursive=True):
        image = load_image(image_file, target_size)
        bary = barycentre(image)
        barycentre_moy += bary
        t += 1

    if t > 0:
        barycentre_moy /= t

    L.append(barycentre_moy)
    print(f"Barycentre moyen pour {direction} : {barycentre_moy}\n")

# Exemple de calcul de seuils automatiques
THRESHOLD_VERTICAL = (np.mean(O2[1]) + np.mean(O2[2]))/2
SEUIL1_droite_milieu = (np.mean(L[3][1]) + np.mean(L[4][1]))/2
SEUIL2_milieu_gauche = (np.mean(L[0][1]) + np.mean(L[1][1]))/2

# print("Seuils automatiques calculés :")
# print(f"THRESHOLD_VERTICAL = {THRESHOLD_VERTICAL}")
# print(f"SEUIL1_droite_milieu = {SEUIL1_droite_milieu}")
# print(f"SEUIL2_milieu_gauche = {SEUIL2_milieu_gauche}")
print("Toutes les captures sont terminées. Bonne session !")




cap.release()
cv2.destroyAllWindows()












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
    
    return ellipse_params




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
            image = load_image(image_path, target_size)  # Fonction pour charger l'image
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






output_file = "features_left_eye.csv"

dataset_path = "dataset/left_eye"

process_dataset_ellipse(dataset_path, output_file)




###########################################################NEURAL NETWORK####################################################  NN#########

###########################################################NEURAL NETWORK####################################################  NN#########

# Charger les données sous forme de tableau numpy
df = pd.read_csv("features_left_eye.csv", sep=';')
data = df.to_numpy()

# Séparer les features (X) et labels (y)
X = data[:, 1:8].astype(float)  # barycenter_x et barycenter_y
y = data[:, 0]  # direction
y_non_encodé = y

# Encoder les labels (directions du regard)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

num_classes = 6
y = to_categorical(y, num_classes)


# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Séparation en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


######################################################################
# Définition du modèle MLP avec 8 caractéristiques en entrée
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(7,)),  # 8 features en entrée
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(6, activation="softmax")  # 6 classes pour la direction du regard
])

# Compilation du modèle avec categorical_crossentropy (pour l'encodage one-hot)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",  # Utiliser categorical_crossentropy pour one-hot
              metrics=["accuracy"])


# Entraînement du modèle
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), callbacks=[early_stopping])

##########################################################
# Prédiction sur les données de test
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)






#################################################################### Nettoyage des images enregistrées après calibration
for direction in directions:
    output_dir = os.path.join(dataset_path, direction)
    for image_file in glob.glob(os.path.join(output_dir, '*.png')):
        os.remove(image_file)



# -------------------------------------------------------------------
# ------------------- FONCTIONS D'ANALYSE EN DIRECT ------------------
# -------------------------------------------------------------------

##############################################################################DELAY BETWEEN DIRECTION CHOSEN########################
NB_FRAMES_ANALYSE = 60


# Petit dictionnaire pour convertir la direction en un index (0..5)
direction_to_coin = {
    "haut_gauche": 0,
    "haut_droite": 1,
    "bas_droite": 2,
    "bas_gauche": 3,
    "haut_milieu": 4,
    "bas_milieu": 5
}





def process_dataset_ellipse_direct(frame, target_size = target_size):
    data_features = []
    #scaler avec mediapipe pour recuperer l'image de l'oeil seul et pas l'image totale renvoyée par la webcam
    # Extraction des paramètres de l'ellipse
    barycenter = barycentre(frame)
    # Traitement
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x_coords = [int(face_landmarks.landmark[idx].x * frame.shape[1]) for idx in left_eye_indices]
            y_coords = [int(face_landmarks.landmark[idx].y * frame.shape[0]) for idx in left_eye_indices]

            x_min = max(0, min(x_coords) - 10)
            x_max = min(frame.shape[1], max(x_coords) + 10)
            y_min = max(0, min(y_coords) - 10)
            y_max = min(frame.shape[0], max(y_coords) + 10)
            eye_region = frame[y_min:y_max, x_min:x_max]
            eye_region = cv2.resize(eye_region, target_size)
            #eye_region = load_image(eye_region,target_size)

            ellipse_params = visualize_eye_detection(eye_region)
            if ellipse_params:
                center, major_axis, minor_axis, angle = ellipse_params
                center_x, center_y = center
                data_features.append([
                    round(float(barycenter[0]),1), round(float(barycenter[1]),1), 
                    round(float(center_x),1), round(float(center_y),1), 
                    round(float(major_axis),1), round(float(minor_axis),1), round(float(angle),1)
                ])
    return data_features

def get_index_of_max(lst):
    return lst.index(max(lst))



import joblib

df = pd.read_csv("features_left_eye.csv", sep=';')

# Enlève la colonne 'direction' (catégorielle)
X = df.drop(columns=['direction'])

# Fit scaler avec les 7 colonnes restantes
scaler = StandardScaler()
scaler.fit(X)

# Sauvegarde si tu veux le réutiliser
joblib.dump(scaler, "scaler_7_features.pkl")




def get_direction_frames(cap, face_mesh, left_eye_indices, NB_FRAMES):
    directions_detectees = []
    for _ in range(NB_FRAMES):
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture caméra.")
            return None

        features = process_dataset_ellipse_direct(frame)
        if features and len(features) > 0:
            columns = ['barycenter_x', 'barycenter_y', 'center_x', 'center_y', 'major_axis', 'minor_axis', 'angle']
            df_features = pd.DataFrame([features[0]], columns=columns)
            features_scaled = scaler.transform(df_features)
            
            # Suppression du log de prédiction
            prediction = model.predict(features_scaled, verbose=0)
            directions_detectees.append(prediction)

        cv2.imshow("Eye Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

        time.sleep(0.01)

    index_max = [get_index_of_max(p.tolist()[0]) for p in directions_detectees]

    counts = Counter(index_max)
    most_common = counts.most_common(1)[0][0]

    # Dictionnaire inverse (index → label)
    index_to_direction = {
        0: "bas_droit",
        1: "bas_gauche",
        2: "haut_droit",
        3: "haut_gauche",
        4: "milieu_bas",
        5: "milieu_haut"
    }

    direction_finale = index_to_direction.get(most_common, "inconnue")

    print(f"Direction dominante détectée sur {NB_FRAMES} frames : {direction_finale}")
    return direction_finale

# # -------------------------------------------------------------------
# # ----------------------- PHASE DE START/STOP ------------------------
# # -------------------------------------------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh_live = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap_live = cv2.VideoCapture(0)
if not cap_live.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
    exit(1)

BTN_WIDTH = 200
BTN_HEIGHT = 80
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

btn_x1 = (WINDOW_WIDTH - BTN_WIDTH) // 2
btn_y1 = (WINDOW_HEIGHT - BTN_HEIGHT) // 2
btn_x2 = btn_x1 + BTN_WIDTH
btn_y2 = btn_y1 + BTN_HEIGHT

capture_active = False

def mouse_callback(event, x, y, flags, param):
    global capture_active
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            capture_active = not capture_active

cv2.namedWindow("Eye Tracking")
cv2.setMouseCallback("Eye Tracking", mouse_callback)

print("Phase d'analyse en direct. Cliquez sur le bouton Start/Stop pour lancer ou mettre en pause.\nAppuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap_live.read()
    if not ret:
        print("Erreur lors de la lecture de la vidéo.")
        break

    # On redimensionne pour 640x480 (optionnel)
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # Dessin du bouton
    btn_color = (0, 255, 0) if not capture_active else (0, 0, 255)
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, -1)
    text = "START" if not capture_active else "STOP"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    text_x = (WINDOW_WIDTH - text_size[0]) // 2
    text_y = (WINDOW_HEIGHT + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

    # On affiche la frame (avec le bouton) en temps réel
    cv2.imshow("Eye Tracking", frame)

    if capture_active:
        # On veut lire DEUX directions successives
        dir1 = get_direction_frames(cap_live, face_mesh_live, left_eye_indices, NB_FRAMES_ANALYSE)
        if dir1 is None:
            # Soit erreur, soit 'q' pressé
            break
        print(f"Première direction : {dir1}")

        dir2 = get_direction_frames(cap_live, face_mesh_live, left_eye_indices, NB_FRAMES_ANALYSE)
        if dir2 is None:
            break
        print(f"Deuxième direction : {dir2}")

        if dir1 in direction_to_coin and dir2 in direction_to_coin:
            coin1 = direction_to_coin[dir1]
            coin2 = direction_to_coin[dir2]
            caractere = feuille_numerique(coin1, coin2)
            print(f"Caractère obtenu : {caractere}")
        else:
            print("Direction(s) inconnue(s), impossible de trouver un caractère.")

    # Gère la sortie au clavier
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap_live.release()
cv2.destroyAllWindows()




