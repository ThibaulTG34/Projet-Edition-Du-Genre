# This Python file uses the following encoding: utf-8
import cv2
import numpy as np
import dlib
import matplotlib
import os

class CNN:
    def __init__(self):
        super().__init__()

    def resize(self):
        pwd = os.path.dirname(__file__)
        dossier_male = pwd+'/Training/male/'
        dossier_female = pwd+'/Training/female/'

        for nom_fichier in os.listdir(dossier_male):
            img = cv2.imread(dossier_male+nom_fichier)
            img = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)