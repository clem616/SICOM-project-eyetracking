import pyautogui
import cv2
import numpy as np

# Dimensions écran


screen_width, screen_height = pyautogui.size()

# Définition des zones
zones = {
    "haut_gauche":    ((0, 0), (screen_width // 3, screen_height // 3)),
    "haut_milieu":    ((screen_width // 3, 0), (2 * screen_width // 3, screen_height // 3)),
    "haut_droite":    ((2 * screen_width // 3, 0), (screen_width, screen_height // 3)),
    "bas_gauche":     ((0, 2 * screen_height // 3), (screen_width // 3, screen_height)),
    "bas_milieu":     ((screen_width // 3, 2 * screen_height // 3), (2 * screen_width // 3, screen_height)),
    "bas_droite":     ((2 * screen_width // 3, 2 * screen_height // 3), (screen_width, screen_height)),
}

zone_to_index = {
    "haut_gauche": 0,
    "haut_milieu": 1,
    "haut_droite": 2,
    "bas_gauche": 3,
    "bas_milieu": 4,
    "bas_droite": 5,
}

def feuille_numerique(coin_choisit_1, coin_choisit_2):
    tableau = [
        ["a", "f", "b", "c", " ", "d"],
        ["g", "k", "h", "i", "suppresseur", "j"],
        ["l", "p", "m", "n", "suppr_mot", "o"],
        ["q", "u", "r", "s", "entrée", "t"],
        ["v", "z", "w", "x", "e", "y"],
        ["1", "2", "3", "4", "5", "."],
    ]
    return tableau[coin_choisit_1][coin_choisit_2]

def get_mouse_zone():
    x, y = pyautogui.position()
    for name, ((x1, y1), (x2, y2)) in zones.items():
        if x1 <= x < x2 and y1 <= y < y2:
            return name
    return None

# Initialisation Interface utilisateur 
cv2.namedWindow("Visual Typing", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Visual Typing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
""" cv2.resizeWindow("Visual Typing", screen_width//2, screen_height)
cv2.moveWindow("Visual Typing", 0, 0) """

text = ""
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
thickness = 3
line_spacing = 60

selection = []

while True:
    frame = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

    # Affichage des zones
    for name, ((x1, y1), (x2, y2)) in zones.items():
        color = (200, 200, 200)
        if get_mouse_zone() == name:
            color = (100, 100, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        label = name.replace("_", "\n")
        cv2.putText(frame, name, (x1 + 10, y1 + 40), font, 0.8, (0, 0, 0), 2)

    # Texte à afficher (centré)
    lines = text.split("\n")
    total_text_height = len(lines) * line_spacing
    start_y = (screen_height - total_text_height) // 2 + 30

    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_x = (screen_width - text_size[0]) // 2
        text_y = start_y + i * line_spacing
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    cv2.imshow("Visual Typing", frame)

    key = cv2.waitKey(100)

    if key == ord("q"):
        break
    elif key == ord(" "):  # ESPACE = valider sélection
        zone = get_mouse_zone()
        if zone:
            selection.append(zone_to_index[zone])
            print(f"Zone sélectionnée : {zone}")
            if len(selection) == 2:
                c = feuille_numerique(selection[0], selection[1])
                if c == "entrée":
                    text += "\n"
                elif c == "suppresseur":
                    text = text[:-1]
                elif c == "suppr_mot":
                    text = " ".join(text.split(" ")[:-1])
                else:
                    text += c
                selection.clear()

        # Sauvegarde
        with open("resultat.txt", "w", encoding="utf-8") as f:
            f.write(text)

cv2.destroyAllWindows()
