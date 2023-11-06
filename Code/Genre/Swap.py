import os
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency, pearsonr
from scipy.spatial import distance as scipy_distance

#Frequences
chi2_weight = 2
#Geometrie positionnelle
euclidiean_weight = 1
#Geometrie forme
pearson_weight = 1
#Texture
bhattacharyya_weight = 0.5

class Swap:
    def __init__(self):
        super().__init__()
        self.face = None
        self.face_name = str(" ")
        self.body = cv2.imread("./trump.jpeg")
        self.body_name = "./trump.jpeg"
        self.w = chi2_weight + euclidiean_weight +  pearson_weight + bhattacharyya_weight

        self.fps = 15 # + vite - vite -> lecture [temps video]
        self.num_frames = 100 # + longue - longue -> interpolation [smoothing video]

    def set_face(self,name):
        self.face_name = str(name)
        self.face = cv2.imread(str(name))

    def set_body(self,name):
        self.body_name = str(name)
        self.body = cv2.imread(str(name))

    def set_img_face(self,f):
        self.face = f

    def set_img_body(self,b):
        self.body = b

    def set_animation(self,name):
        self.in_animation = str(name)

    def set_frames(self, f):
        self.num_frames = int(f)

    def set_fps(self, f):
        self.fps = int(f)

    def prepare_variables(self):
        self.face_gray = cv2.cvtColor(self.face, cv2.COLOR_BGR2GRAY)
        self.body_gray = cv2.cvtColor(self.body, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.face_gray.shape
        self.mask = np.zeros((self.height, self.width), np.uint8)
        self.height, self.width, self.channels = self.body.shape
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./shape_predictor_81_face_landmarks.dat")

    def get_landmarks(self, landmarks, landmarks_points):
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))

    def get_index(self, arr):
        index = 0
        if arr[0]:
            index = arr[0][0]
        return index

    def init_landmark(self):
        self.rect = self.detector(self.face_gray)[0]
        self.landmarks = self.predictor(self.face_gray, self.rect)
        self.landmarks_points = []
        self.get_landmarks(self.landmarks, self.landmarks_points)
        self.points = np.array(self.landmarks_points, np.int32)

    def init_contour(self):
        self.contour = cv2.convexHull(self.points)
        self.face_cp = self.face.copy()
        self.face_image_1 = cv2.bitwise_and(self.face, self.face, mask=self.mask)

    def init_maillage(self):
        self.rect = cv2.boundingRect(self.contour)

        self.subdiv = cv2.Subdiv2D(self.rect)
        self.subdiv.insert(self.landmarks_points)
        self.triangles = self.subdiv.getTriangleList()
        self.triangles = np.array(self.triangles, dtype=np.int32)

        self.face_cp = self.face.copy()
        self.indexes_triangles = []

        for triangle in self.triangles :
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])

            cv2.line(self.face_cp, pt1, pt2, (255, 255, 255), 3,  0)
            cv2.line(self.face_cp, pt2, pt3, (255, 255, 255), 3,  0)
            cv2.line(self.face_cp, pt3, pt1, (255, 255, 255), 3,  0)

            index_pt1 = np.where((self.points == pt1).all(axis=1))
            index_pt1 = self.get_index(index_pt1)
            index_pt2 = np.where((self.points == pt2).all(axis=1))
            index_pt2 = self.get_index(index_pt2)
            index_pt3 = np.where((self.points == pt3).all(axis=1))
            index_pt3 = self.get_index(index_pt3)

            if (index_pt1 is not None) and (index_pt2 is not None) and (index_pt3 is not None):
                vertices = [index_pt1, index_pt2, index_pt3]
                self.indexes_triangles.append(vertices)

    def config_face(self):
        self.rect2 = self.detector(self.body_gray)[0]
        self.landmarks_2 = self.predictor(self.body_gray, self.rect2)
        self.landmarks_points2 = []
        self.get_landmarks(self.landmarks_2, self.landmarks_points2)
        self.points2 = np.array(self.landmarks_points2, np.int32)
        self.contour2 = cv2.convexHull(self.points2)
        self.body_cp = self.body.copy()

    def get_face(self):
        self.lines_space_new_face = np.zeros((self.height, self.width, self.channels), np.uint8)
        self.body_new_face = np.zeros((self.height, self.width, self.channels), np.uint8)
        self.height, self.width = self.face_gray.shape
        self.lines_space_mask = np.zeros((self.height, self.width), np.uint8)

    def traitement(self):
        for triangle in self.indexes_triangles:
            ################################################################################
            #*****************************************
            pt1 = self.landmarks_points[triangle[0]]
            pt2 = self.landmarks_points[triangle[1]]
            pt3 = self.landmarks_points[triangle[2]]
            #*****************************************

            (x, y, widht, height) = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
            cropped_triangle = self.face[y: y+height, x: x+widht]
            cropped_mask = np.zeros((height, widht), np.uint8)

            points = np.array([[pt1[0]-x, pt1[1]-y], [pt2[0]-x, pt2[1]-y], [pt3[0]-x, pt3[1]-y]], np.int32)
            cv2.fillConvexPoly(cropped_mask, points, 255)

            cv2.line(self.lines_space_mask, pt1, pt2, 255)
            cv2.line(self.lines_space_mask, pt2, pt3, 255)
            cv2.line(self.lines_space_mask, pt1, pt3, 255)

            lines_space = cv2.bitwise_and(self.face, self.face, mask=self.lines_space_mask)
            ################################################################################
            #******************************************
            pt1 = self.landmarks_points2[triangle[0]]
            pt2 = self.landmarks_points2[triangle[1]]
            pt3 = self.landmarks_points2[triangle[2]]
            #******************************************

            (x, y, widht, height) = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
            cropped_mask2 = np.zeros((height,widht), np.uint8)

            points2 = np.array([[pt1[0]-x, pt1[1]-y], [pt2[0]-x, pt2[1]-y], [pt3[0]-x, pt3[1]-y]], np.int32)
            cv2.fillConvexPoly(cropped_mask2, points2, 255)

            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            points =  np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            dist_triangle = cv2.warpAffine(cropped_triangle, M, (widht, height))
            dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=cropped_mask2)

            body_new_face_rect_area = self.body_new_face[y: y+height, x: x+widht]
            body_new_face_rect_area_gray = cv2.cvtColor(body_new_face_rect_area, cv2.COLOR_BGR2GRAY)

            masked_triangle = cv2.threshold(body_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=masked_triangle[1])

            body_new_face_rect_area = cv2.add(body_new_face_rect_area, dist_triangle)
            self.body_new_face[y: y+height, x: x+widht] = body_new_face_rect_area


    def replace(self):
        self.body_face_mask = np.zeros_like(self.body_gray)
        self.body_head_mask = cv2.fillConvexPoly(self.body_face_mask, self.contour2, 255)
        self.body_face_mask = cv2.bitwise_not(self.body_head_mask)
        self.body_maskless = cv2.bitwise_and(self.body, self.body, mask=self.body_face_mask)
        self.result = cv2.add(self.body_maskless, self.body_new_face)

    def smoothing(self):
        (x, y, widht, height) = cv2.boundingRect(self.contour2)
        self.center_face2 = (int((x+x+widht)/2), int((y+y+height)/2))
        self.seamlessclone = cv2.seamlessClone(self.result, self.body, self.body_head_mask, self.center_face2, cv2.NORMAL_CLONE)
        self.result = self.seamlessclone

    def chi2_distance(self, hist1, hist2):
        chi2 = chi2_contingency([hist1, hist2])
        return chi2

    def swap(self):
        if self.face is None:
            image_extensions = [".jpg", ".jpeg", ".png"]
            predictor = dlib.shape_predictor("./shape_predictor_81_face_landmarks.dat")
            detector = dlib.get_frontal_face_detector()
            min_distance = float('inf')
            closest_face = None
            _name = str(" ")

            for file_name in os.listdir("./"):
                if file_name.lower().endswith(tuple(image_extensions)):
                    image_path = os.path.join("./", file_name)
                    image = cv2.imread(image_path)

                    if image is not None:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        faces = detector(gray)

                        gray_reference = cv2.cvtColor(self.body, cv2.COLOR_BGR2GRAY)
                        faces_reference = detector(gray_reference)

                        if len(faces) > 0:
                            body_landmarks = np.array([[p.x, p.y] for p in predictor(self.body, faces_reference[0]).parts()])
                            landmarks = np.array([[p.x, p.y] for p in predictor(image, faces[0]).parts()])

                            #Frequences
                            chi2_distance = float(self.chi2_distance(body_landmarks.flatten(), landmarks.flatten()).statistic)
                            #Geometrie
                            euclidean_distance = float(np.linalg.norm(body_landmarks.flatten() - landmarks.flatten()))
                            #Forme
                            shape_correlation = float(pearsonr(body_landmarks.flatten(), landmarks.flatten()).statistic)
                            #Texture
                            hist1 = cv2.calcHist([self.body], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                            hist2 = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                            texture_distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

                            d = (chi2_weight * chi2_distance + euclidiean_weight * euclidean_distance + pearson_weight * shape_correlation + bhattacharyya_weight * texture_distance)

                            distance = float(d / self.w)

                            file__name, file_extension = os.path.splitext(os.path.basename(file_name))
                            body__name, body_extension = os.path.splitext(os.path.basename(self.body_name))

                            if distance < min_distance and not (file__name == body__name):
                                min_distance = distance
                                closest_face = image
                                _name = image_path

            self.finded_face = str(_name)
            self.face = closest_face
            print("Auto Selection face : " + self.finded_face)

        self.prepare_variables()
        self.init_landmark()
        self.init_contour()
        self.init_maillage()
        self.config_face()
        self.get_face()
        self.traitement()
        self.replace()
        self.smoothing()

    def get_animation(self):
        #file__name, file_extension = os.path.splitext(os.path.basename(str(self.body_name)))
        #self.in_animation = str(file__name + "_anim")
        self.animation()

    def get_result(self):
        return self.result

    def save(self, name=None):
        res_filename = str()
        if name is None:
            res_filename = str("./result.png")
        else:
            res_filename = str(name)
        cv2.imwrite(str(filename), self.result)

    def animation(self):
        if self.body.shape != self.result.shape:
            raise ValueError("Les deux images doivent avoir la même taille.")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.in_animation, fourcc, self.fps, (self.body.shape[1], self.body.shape[0]))

        for i in range(self.num_frames):
            alpha = i / (self.num_frames - 1)
            interpolated_image = cv2.addWeighted(self.body, 1 - alpha, self.result, alpha, 0)
            out.write(interpolated_image)

        print("Animation created in : " + self.in_animation)
        self.out_animation = self.in_animation
        out.release()

