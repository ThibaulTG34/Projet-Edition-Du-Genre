# This Python file uses the following encoding: utf-8
import argparse
import imutils
from imutils import face_utils
import cv2
import numpy as np
import dlib
import matplotlib

class Morph:
    def __init__(self):
        pass

    def init_landmark(self):

        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-p", "–shape-predictor", required=True, help="path to facial landmark predictor")
        self.ap.add_argument("-i", "–image", required=True, help="path to input image")
        self.args = vars(self.ap.parse_args())

        self.detector = dlib.get_frontal_face_detector()
        if(self.args["shape_predictor"] is not None) : self.predictor = dlib.shape_predictor(self.args["shape_predictor"])
        else : self.predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

        self.image = cv2.imread(self.args["image"])
        self.image = imutils.resize(self.image, width=500)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.rects = detector(self.gray, 1)

    def detect_landmark(self):

        for (i, rect) in enumerate(self.rects):
            self.shape = predictor(self.gray, rect)
            self.shape = face_utils.shape_to_np(self.shape)
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(self.image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for (x, y) in self.shape:
                cv2.circle(self.image, (x, y), 1, (0, 0, 255), -1)

        # Show
        cv2.imshow("Output", self.image)
        cv2.waitKey(0)

