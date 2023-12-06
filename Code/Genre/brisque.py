import cv2 as cv
import sys

def calculate_scores(image_path):
    img = cv.imread(image_path)
    if img is None:
        print(f"Impossible de lire l'image à partir du chemin : {image_path}")
        return

    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurScore = cv.Laplacian(grey, cv.CV_64F).var()
    score = cv.quality.QualityBRISQUE_compute(img, "./classifieurs/brisque_model_live.yml", "./classifieurs/brisque_range_live.yml")

    print(f' >> Blur Score: {blurScore}')
    print(f' >> BRISQUE Score: {score}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_image>")
    else:
        image_path = sys.argv[1]
        calculate_scores(image_path)
