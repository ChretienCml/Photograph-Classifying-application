# -*- coding: utf-8 -*-


'''
Main of the recognition face programm

Author : Camille CHRETIEN
maj : 11/07/2018
'''

import cv2

# to make it dynamic with Qt
image_path = "C:/Users/Solange/Desktop/Documents/Hackiflette/100-days_code/face_recognition/"
image_name = "test.JPG"

face_cascade_path = "haarcascade_frontalface_default.xml"

scale_factor = 1.1
min_square = 5

print("Opening :", image_name)
photograph_to_analyse = cv2.imread(image_path + image_name)
photograph_to_analyse_width = photograph_to_analyse.shape[1]
photograph_to_analyse_heigh = photograph_to_analyse.shape[0]
print("Width :", photograph_to_analyse_width, "Heigh :", photograph_to_analyse_heigh)

if photograph_to_analyse != None:
    photograph_to_analyse = cv2.resize(photograph_to_analyse,
                                      (photograph_to_analyse_width / 10, photograph_to_analyse_heigh / 10))

    cv2.imshow("1", photograph_to_analyse)
    cv2.waitKey(0)

    print("Loading Haar Cascade")
    haar_cascade_face = cv2.CascadeClassifier(face_cascade_path)

    print("Converting to grey")
    grey_photograph_to_analyse = cv2.cvtColor(photograph_to_analyse, cv2.COLOR_BGR2GRAY)

    print("Searching for face")
    faces = haar_cascade_face.detectMultiScale(grey_photograph_to_analyse,
                                               scale_factor,
                                               min_square)

    if (len(faces) != 0):
        print("Faces found !")
        for (x, y, w, h) in faces:
            cv2.rectangle(photograph_to_analyse, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Faces found", photograph_to_analyse)
        cv2.waitKey(0)

print("Runtime OK !")
