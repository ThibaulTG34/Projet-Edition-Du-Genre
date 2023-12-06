import os
import cv2 
import matplotlib.pyplot as plt

# Image
path = './datasets/genderchange/train/B/woman_75.jpg'  

def classified(path):
    if os.path.isfile(path):
        image = cv2.imread(path)

        if (image is None) or (image.size == 0):
            print("classified as Unknow")
            return str("Unknow")

        image = cv2.resize(image, (720, 640))
        fr_cv = image.copy()
        
        pwd = os.path.dirname(__file__)

        # Importing Models and set mean values 
        face1 = pwd+"/classifieurs/opencv_face_detector.pbtxt"
        face2 = pwd+"/classifieurs/opencv_face_detector_uint8.pb"
        gen1 = pwd+"/classifieurs/gender_deploy.prototxt"
        gen2 = pwd+"/classifieurs/gender_net.caffemodel"
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  

        # Using models 
        # Face 
        face = cv2.dnn.readNet(face2, face1) 
        # gender 
        gen = cv2.dnn.readNet(gen2, gen1) 

        if face.empty():
            print("classified as Unknow")
            return str("Unknow")

        if gen.empty():
            print("classified as Unknow")
            return str("Unknow")

        # Category of distribution 
        lg = ['Male', 'Female'] 
                
        # Face detection 
        fr_h = fr_cv.shape[0] 
        fr_w = fr_cv.shape[1] 
        try:
            blob = cv2.dnn.blobFromImage(
                fr_cv, 1.0, (300, 300), [104, 117, 123], True, False)
        except cv2.error as e:
            print(f"Une erreur s'est produite lors de la création du blob : {e}")
            return str("Unknow")
                
        face.setInput(blob) 
        detections = face.forward()
                            
        # Face bounding box creation 
        faceBoxes = [] 
        for i in range(detections.shape[2]): 
            
            confidence = detections[0, 0, i, 2] 
            if confidence > 0.7: 
                        
                x1 = int(detections[0, 0, i, 3]*fr_w) 
                y1 = int(detections[0, 0, i, 4]*fr_h) 
                x2 = int(detections[0, 0, i, 5]*fr_w) 
                y2 = int(detections[0, 0, i, 6]*fr_h) 
                
                faceBoxes.append([x1, y1, x2, y2]) 
                
                cv2.rectangle(fr_cv, (x1, y1), (x2, y2), 
                            (0, 255, 0), int(round(fr_h/150)), 8) 

        if not faceBoxes: 
            print("classified as Unknow")
                
        #Extracting face 
        face = fr_cv[max(0, faceBoxes[0][1]-15): 
                    min(faceBoxes[0][3]+15, fr_cv.shape[0]-1), 
                    max(0, faceBoxes[0][0]-15):min(faceBoxes[0][2]+15, 
                                fr_cv.shape[1]-1)] 
            
        #Extracting the main blob 
        try:
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        except cv2.error as e:
            print(f"Une erreur s'est produite lors de la création du blob : {e}")
            return str("Unknow")

        #Prediction of gender 
        gen.setInput(blob) 
        genderPreds = gen.forward() 
        gender = lg[genderPreds[0].argmax()] 
            
        #print(f'classified as {gender}')
        return str(gender)

#classified(path)
