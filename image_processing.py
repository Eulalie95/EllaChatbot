from PIL import Image
import pytesseract

def extract_text_from_image(image_path):
    """
    Extrait le texte d'une image en utilisant pytesseract.

    Args:
        image_path (str): Le chemin vers le fichier image.

    Returns:
        str: Le texte extrait de l'image, ou None en cas d'erreur.
    """
    try:
        # Ouvrir l'image en utilisant Pillow (PIL)
        img = Image.open(image_path)

        # Utiliser pytesseract pour extraire le texte
        text = pytesseract.image_to_string(img, lang='fra') # Spécifiez la langue si nécessaire ('fra' pour le français)

        return text
    except Exception as e:
        print(f"Erreur lors de l'extraction du texte de l'image '{image_path}': {e}")
        return None
