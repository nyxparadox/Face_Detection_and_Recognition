# import----------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2 as cv
import numpy as np
import pickle
from keras_facenet import FaceNet


from mtcnn.mtcnn import MTCNN

embedder = FaceNet()

def Get_Embedding(face_img):
    face_img = face_img.astype('float32')  #3D (160x160x3)
    face_img = np.expand_dims(face_img, axis=0)  #4D (Nonex160x160x3)
    yhat = embedder.embeddings(face_img)
    return yhat[0] 

detector = MTCNN()
t_im = cv.imread("___Gave/Image/Path ___")   # provide image path with image name (e.g: /home/pictures/my_pic/unknown.jpg)
Displ_img = t_im.copy()
t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
x,y,w,h = detector.detect_faces(t_im)[0]['box']

t_im = t_im[y:y+h, x:x+w]
t_im = cv.resize(t_im, (160,160))
test_im = Get_Embedding(t_im)


#-----------------------------------------
with open("face_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["svm"]
encoder = artifacts["encoder"]

#------------------------------------------

test_im =[test_im]
ypreds= model.predict(test_im)
person = encoder.inverse_transform(ypreds)


# displaying image with recognization
cv.rectangle(Displ_img ,(x,y), (x+w,y+h), (0,255,0), 2)
cv.putText(Displ_img, str(person[0]), (x,y-10), cv.FONT_HERSHEY_COMPLEX, 1.1, (0,255,0), 2)
cv.imshow("FACE RECOGNIZER WINDOW", Displ_img)
cv.waitKey(0)
cv.destroyAllWindows()

print(f"name value_index: {ypreds}")
print(f"Identified Person: {person}")



