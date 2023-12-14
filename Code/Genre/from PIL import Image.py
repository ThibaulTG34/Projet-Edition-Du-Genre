from PIL import Image
import os

def inverser_couleurs(image_path, save_path):
    # Charger l'image
    image = Image.open(image_path)

    # Convertir l'image en noir et blanc
    image_noir_blanc = image.convert("L")

    # Inverser les couleurs
    image_inversee = Image.eval(image_noir_blanc, lambda x: 255 - x)

    # Sauvegarder l'image inversée
    image_inversee.save(save_path)

pwd = os.path.dirname(__file__)
# Exemple d'utilisation
image_path = pwd+"/img.jpg"
save_path = pwd+"/image_inversee.jpg"

inverser_couleurs(image_path, save_path)
