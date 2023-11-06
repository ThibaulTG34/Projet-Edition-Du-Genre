# This Python file uses the following encoding: utf-8
import argparse
import imutils
import cv2
import numpy as np
import dlib
import matplotlib

class Morph:
    def __init__(self):
        super().__init__()
        self.target = cv2.imread("madona.jpeg")
        self.source = cv2.imread("trump.jpeg")
        self.power = float(0.5)
        self.init_landmark()

    def set_name(self, name):
        self.filename = str(name)

    def set_source(self, cible):
        self.source = cv2.imread(str(cible))

    def set_target(self, target):
        self.target = cv2.imread(str(target))

    def set_power(self, power):
        self.power = float(power)

    def init_landmark(self):
        self.predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()

    def init_sources(self):
        self.source_faces = self.detector(self.source)
        if not self.source_faces:
            raise ValueError("Aucun visage détecté dans l'image source.")

        self.target_faces = self.detector(self.target)
        if not self.target_faces:
            raise ValueError("Aucun visage détecté dans l'image cible.")

    def set_landmark(self):
        self.source_landmarks = self.predictor(self.source, self.source_faces[0])
        self.target_landmarks = self.predictor(self.target, self.target_faces[0])

    def interpolate(self, option = 1):
        self.morphed_landmarks = []
        if option == 1:
            for source_point, target_point in zip(self.source_landmarks.parts(), self.target_landmarks.parts()):
                x = int((1 - self.power) * source_point.x + self.power * target_point.x)
                y = int((1 - self.power) * source_point.y + self.power * target_point.y)
                self.morphed_landmarks.append((x, y))
        self.morphed_image = cv2.addWeighted(self.source, 1 - self.power, self.target, self.power, 0)

    def apply(self):
        self.source_copy = self.source.copy()
        for x, y in self.morphed_landmarks:
            cv2.circle(self.source_copy, (x, y), 3, (0, 255, 0), -1)

        self.landmark = self.source.copy
        cv2.waitKey(0)

    def save(self, name = None):
        if name is None:
            cv2.imwrite("morphed_face.jpg", source_copy)
        else:
            cv2.imwrite(str(name), source_copy)

    def get_result(self):
        return self.result

    def clear(self):
        cv2.destroyAllWindows()

    def linear_morph(self):
        if self.source is None or self.target is None:
            raise ValueError("Impossible de charger les images source ou cible.")

        if self.source.shape != self.target.shape:
            source_height, source_width, _ = self.source.shape
            target_height, target_width, _ = self.target.shape

            self.max_height = max(source_height, target_height)
            self.max_width = max(source_width, target_width)

            self.source = cv2.resize(self.source, (self.max_width, self.max_height))
            self.target = cv2.resize(self.target, (self.max_width, self.max_height))

        self.result = cv2.addWeighted(self.source, 1 - self.power, self.target, self.power, 0)

    def linear_morph2(self) :
        if self.source is None or self.target is None:
                raise ValueError("Impossible de charger les images source ou cible.")

        source_height, source_width, _ = self.source.shape
        target_height, target_width, _ = self.target.shape
        height_ratio = target_height / source_height
        width_ratio = target_width / source_width

        if height_ratio > width_ratio:
            new_height = target_height
            new_width = int(source_width * height_ratio)
        else:
            new_width = target_width
            new_height = int(source_height * width_ratio)

        source_resized = cv2.resize(self.source, (new_width, new_height))
        target_resized = cv2.resize(self.target, (new_width, new_height))

        self.result = cv2.addWeighted(source_resized, 1 - self.power, target_resized, self.power, 0)

    def predict_landmarks(self, image):
        faces = self.detector(image)
        if not faces:
            raise ValueError("Aucun visage détecté dans l'image.")

        shape = self.predictor(image, faces[0])
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]

        return landmarks

    def interpolate_landmarks(self):
        self.source_landmarks = self.predict_landmarks(self.source)
        self.target_landmarks = self.predict_landmarks(self.target)

        self.interpolated_landmarks = []
        for source_point, target_point in zip(self.source_landmarks, self.target_landmarks):
            interpolated_x = int((1 - self.power) * source_point[0] + self.power * target_point[0])
            interpolated_y = int((1 - self.power) * source_point[1] + self.power * target_point[1])
            self.interpolated_landmarks.append((interpolated_x, interpolated_y))

        self.interpolated_landmarks = np.array(self.interpolated_landmarks)
        #cv2.addWeighted(self.source, 1 - self.power, self.target, self.power, 0)

    def landmark_morph(self):
        if self.source is None or self.target is None:
            raise ValueError("Impossible de charger les images.")

        self.init_landmark()
        self.init_sources()
        self.set_landmark()
        self.source_shape = self.source.shape
        self.target_shape = self.target.shape
        self.interpolate_landmarks()

        if self.source_shape != self.target_shape:
            self.target = cv2.resize(self.target, (self.source_shape[1], self.source_shape[0]))

        y, x = np.mgrid[0:self.source_shape[0], 0:self.source_shape[1]]

        for source_point, target_point, interpolated_point in zip(self.source_landmarks, self.target_landmarks, self.interpolated_landmarks):
                source_x, source_y = source_point
                target_x, target_y = target_point
                interpolated_x, interpolated_y = interpolated_point

                x[source_y, source_x] = x[interpolated_y, interpolated_x]
                y[source_y, source_x] = y[interpolated_y, interpolated_x]

        self.result = cv2.remap(self.target, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def landmark(self):
        self.init_landmark()
        self.init_sources()
        self.set_landmark()
        self.interpolate()
        self.apply()
