
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

# -------------------------------------------------------------------
# ------------------------- PHASE DE CALIBRATION ---------------------
# -------------------------------------------------------------------

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

root_output_dir = "/home/clem/Documents/my-eye-tracking/dataset_clem_9/"

# Indices des landmarks des yeux
left_eye_indices = [33, 133, 157, 158, 159, 160, 161, 144, 145, 153, 154, 155]

right_eye_indices = [362, 398, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385]

duration = 5 # secondes
fps = 30
interval = 1 / fps

def load_image(image_path, target_size):
    image = Image.open(image_path)
    image = image.convert('RGB')  
    image = image.resize(target_size)
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

###################### Feuille numérique (inchangé) ########################
# def verso_feuille(coin_choisit_1, coin_choisit_2):
#     if coin_choisit_1 == 0:
#         if coin_choisit_2 == 0:   caractere = "0"
#         elif coin_choisit_2 == 1: caractere = "1"
#         elif coin_choisit_2 == 2: caractere = "2"
#         elif coin_choisit_2 == 3: caractere = "3"
#         elif coin_choisit_2 == 4: caractere = "4"
#         elif coin_choisit_2 == 5: caractere = "("
#     elif coin_choisit_1 == 1:
#         if coin_choisit_2 == 0: caractere = "5"
#         elif coin_choisit_2 == 1: caractere = "6"
#         elif coin_choisit_2 == 2: caractere = "7"
#         elif coin_choisit_2 == 3: caractere = "8"
#         elif coin_choisit_2 == 4: caractere = "9"
#         elif coin_choisit_2 == 5: caractere = ")"
#     elif coin_choisit_1 == 2:
#         if coin_choisit_2 == 0: caractere = "."
#         elif coin_choisit_2 == 1: caractere = "?"
#         elif coin_choisit_2 == 2: caractere = "!"
#         elif coin_choisit_2 == 3: caractere = ","
#         elif coin_choisit_2 == 4: caractere = "on"
#         elif coin_choisit_2 == 5: caractere = "off"
#     elif coin_choisit_1 == 3:
#         if coin_choisit_2 == 0 : caractere = "-"
#         elif coin_choisit_2 == 1: caractere = "+"
#         elif coin_choisit_2 == 2: caractere = "*"
#         elif coin_choisit_2 == 3: caractere = "/"
#         elif coin_choisit_2 == 4: caractere = "="
#         elif coin_choisit_2 == 5: caractere = "%"
#     elif coin_choisit_1 == 4:
#         if coin_choisit_2 == 0:   caractere = "mot1"
#         elif coin_choisit_2 == 1: caractere = "mot2"
#         elif coin_choisit_2 == 2: caractere = "mot3"
#         elif coin_choisit_2 == 3: caractere = "mot4"
#         elif coin_choisit_2 == 4: caractere = "mot5"
#         elif coin_choisit_2 == 5: caractere = "mot6"
#     elif coin_choisit_1 == 5:
#         caractere = feuille_numerique(coin_choisit_1, coin_choisit_2)
#     return caractere




# Liste des directions à capturer
# directions = [
#     "milieu_haut",
#     "haut_gauche",
#     "bas_gauche",
#     "milieu_bas",
#     "bas_droit",
#     "haut_droit"
# ]


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
# ------------------------ DÉBUT CALIBRAGE -----------------------
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
print(L)
print(L[4])
# Exemple de calcul de seuils automatiques
THRESHOLD_VERTICAL = (np.mean(O2[1]) + np.mean(O2[2]))/2
# SEUIL1_droite_milieu = (np.mean(L[3][1]) + np.mean(L[4][1]))/2
# SEUIL2_milieu_gauche = (np.mean(L[0][1]) + np.mean(L[1][1]))/2

SEUIL1_droite_milieu = (np.mean(L[3][1]) + np.mean(L[4][1]))/2  -2
SEUIL2_milieu_gauche = (np.mean(L[2][1]) + np.mean(L[3][1]))/2   +2

print("Seuils automatiques calculés :")
print(f"THRESHOLD_VERTICAL = {THRESHOLD_VERTICAL}")
print(f"SEUIL1_droite_milieu = {SEUIL1_droite_milieu}")
print(f"SEUIL2_milieu_gauche = {SEUIL2_milieu_gauche}")
print("Toutes les captures sont terminées. Bonne session !")

cap.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------------
# ------------------- FONCTIONS D'ANALYSE EN DIRECT ------------------
# -------------------------------------------------------------------
NB_FRAMES_ANALYSE = 60

def determiner_vertical(y_min, y_max, threshold=THRESHOLD_VERTICAL):
    vertical_opening = abs(y_max - y_min)
    return "haut" if vertical_opening > threshold else "bas"

def determiner_horizontal(bary, s1=SEUIL1_droite_milieu, s2=SEUIL2_milieu_gauche):
    x_bary = bary[1]
    if x_bary < s1:
        return "droite"
    elif s1 <= x_bary <= s2:
        return "milieu"
    else:
        return "gauche"

def vote_majoritaire(directions_list):
    compteur = {}
    for d in directions_list:
        compteur[d] = compteur.get(d, 0) + 1
    best_direction = max(compteur, key=compteur.get)
    return best_direction

# Petit dictionnaire pour convertir la direction en un index (0..5)
# Ajustez l'ordre si nécessaire pour qu'il soit cohérent avec verso_feuille
# direction_to_coin = {
#     "haut_gauche": 0,
#     "haut_droite": 1,
#     "bas_droite": 2,
#     "bas_gauche": 3,
#     "haut_milieu": 4,
#     "bas_milieu": 5
# }


direction_to_coin = {
    "haut_milieu": 0,
    "haut_gauche": 1,
    "bas_gauche": 2,
    "bas_milieu": 3,
    "bas_droite": 4,
    "haut_droite": 5
}



def get_direction_30_frames(cap, face_mesh, left_eye_indices, NB_FRAMES):
    """
    Lit NB_FRAMES images de la webcam 'cap', calcule la direction 
    majoritaire via vote_majoritaire, et la renvoie (ex: "haut_gauche").
    Renvoie None si on ne détecte rien ou si on a appuyé sur 'q'.
    """
    directions_detectees = []

    for _ in range(NB_FRAMES):
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture caméra.")
            return None

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

                vertical_dir = determiner_vertical(y_min, y_max, THRESHOLD_VERTICAL)

                eye_region = frame[y_min:y_max, x_min:x_max]
                if eye_region.size > 0:
                    bary = barycentre(eye_region)
                    horizontal_dir = determiner_horizontal(bary, SEUIL1_droite_milieu, SEUIL2_milieu_gauche)
                else:
                    horizontal_dir = "milieu"

                direction_frame = f"{vertical_dir}_{horizontal_dir}"
                directions_detectees.append(direction_frame)

        cv2.imshow("Eye Tracking", frame)
        # Si on appuie sur 'q', on arrête
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

        time.sleep(0.01)

    if len(directions_detectees) == 0:
        return None
    return vote_majoritaire(directions_detectees)








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
        dir1 = get_direction_30_frames(cap_live, face_mesh_live, left_eye_indices, NB_FRAMES_ANALYSE)
        if dir1 is None:
            # Soit erreur, soit 'q' pressé
            break
        print(f"Première direction : {dir1}")

        dir2 = get_direction_30_frames(cap_live, face_mesh_live, left_eye_indices, NB_FRAMES_ANALYSE)
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



