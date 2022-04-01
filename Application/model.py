#!/usr/bin/env python
# coding: utf-8
import pywt
import cv2
import matplotlib.pyplot as cplt
import numpy as np
import pickle, json
import base64



class Classification :
        # roi ---> Region Of Interest
        # class instance
    face_cascade = cv2.CascadeClassifier('dependencies/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('dependencies/opencv/data/haarcascades/haarcascade_eye.xml')    
    
    # for Loading a model from the pickle file
    def load_model(self) :
        with open('dependencies/logisticregression_model.pkl', 'rb') as model :
            self.logistic_regression = pickle.load(model) 
            return self

    # For the Encoding of the y
    def label_encoder(self) :
        self.new_encoding = {

        }
        with open('dependencies/label_encoding_y.json', 'rb') as file :
            y_label_encoding = json.load(file)
           # print(y_label_encoding)
            for key, value in y_label_encoding.items() :
                self.new_encoding[value] = key 
        return self

    # It will convert the decoded base64 byts into array
    def base64_to_image(self, byts) :
        nparr = np.frombuffer(byts, np.uint8)
        print(nparr.shape, "aa")
        self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self
    
    # Without this function our base64 to image won;t work
    # when we implemented the frontend we need to just use decode of base64 method and decode it
    def reading_base664(self, file) :
        with open(file, 'rb') as f :
            f = f.read()
            picture = base64.b64encode(f) 
        with open('new_byts.txt', 'wb') as file :
            file.write(picture) #
        with open('new_byts.txt', 'rb') as file :
            f = file.read()
           # print(self.f)
            self.byts = base64.b64decode(f)
            return self 


    def base64_to_imagee(self, byts) :
        with open("imageToSave.png", "wb") as fh:
            fh.write(base64.decodebytes(byts))

    def wavelet_transformation(self, image, mode = 'haar', level = 1):
        imArray = image
        #Datatype conversions
        #convert to grayscale
        imArray = cv2.cvtColor( imArray, cv2.COLOR_RGB2GRAY )
    #convert to float=
        imArray  =  np.float32(imArray)   
        imArray /= 255
        # compute coefficients 
        coeffs=pywt.wavedec2(imArray, mode, level=level)
        #Process Coefficients
        coeffs_H = list(coeffs)  
        coeffs_H[0] *= 0;  
        # reconstruction
        imArray_H=pywt.waverec2(coeffs_H, mode)
        imArray_H *= 255
        self.imArray_H =  np.uint8(imArray_H)
        return self.imArray_H
    
    def get_cropped_image(self, image):
        img = image
        self.cropped_faces = [ ]
        if img is  not None :
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = Classification.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces: 
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = Classification.eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) >= 2:
                    self.cropped_faces.append(roi_color)
        return self.cropped_faces

            
    def main(self, images) :
        result = []
        # Required Objects for our model . 
        logistic_regression = self.load_model().logistic_regression
        new_label = self.label_encoder().new_encoding
        for img in self.get_cropped_image(images) :          
            print(img, "ana")  
            scalled_raw_img = cv2.resize(img, (32, 32))
            img_har = self.wavelet_transformation(img, 'db1', 5)
            scalled_img_har = cv2.resize(img_har, (32, 32))
            combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
            X = np.array(combined_img)
            X = X.reshape(1, 32*32*3 + 32*32)
            X.astype('float')
            prediction = logistic_regression.predict(X)
            pred_prob = logistic_regression.predict_proba(X)
            answer = new_label[prediction[0]]
            result.append(answer)
        print(result)
        if len(result) >= 1  :

            return result, prediction[0], pred_prob
        return result
