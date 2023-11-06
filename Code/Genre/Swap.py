import os
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt

class Swap:
    def __init__(self):
        super().__init__()
        self.face = cv2.imread("madona.jpeg")
        self.body = cv2.imread("trump.jpeg")

    def set_face(self,name):
        self.face = cv2.imread(str(name))

    def set_body(self,name):
        self.body = cv2.imread(str(name))

    def set_img_face(self,f):
        self.face = f

    def set_img_body(self,b):
        self.body = b

    def prepare_variables(self):
        self.face_gray = cv2.cvtColor(self.face, cv2.COLOR_BGR2GRAY)
        self.body_gray = cv2.cvtColor(self.body, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.face_gray.shape
        self.mask = np.zeros((self.height, self.width), np.uint8)
        self.height, self.width, self.channels = self.body.shape
        self.detector = dlib.get_frontal_face_detector()
        pwd = os.path.dirname(__file__)
        self.predictor = dlib.shape_predictor(pwd + "/shape_predictor_81_face_landmarks.dat")

    def get_landmarks(self, landmarks, landmarks_points):
        for n in range(81):
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

    def swap(self):
        self.prepare_variables()
        self.init_landmark()
        self.init_contour()
        self.init_maillage()
        self.config_face()
        self.get_face()
        self.traitement()
        self.replace()
        self.smoothing()

    def get_result(self):
        return self.result

    def save(self, name=None):
        res_filename = str()
        if name is None:
            res_filename = str("./result.png")
        else:
            res_filename = str(name)
        cv2.imwrite(str(filename), self.result)

