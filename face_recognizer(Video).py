# import libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2 as cv
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet


# initialize
with open("face_artifacts.pkl", 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['svm']
encodder = artifacts['encoder']

embedder = FaceNet()
detector = MTCNN()

# Function 
def Get_Embedding(face_img):
    face_img = face_img.astype('float32')  #3D (160x160x3)
    face_img = np.expand_dims(face_img, axis=0)  #4D (Nonex160x160x3)
    yhat = embedder.embeddings(face_img)
    return yhat[0]



# While Loop
cap = cv.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv.resize(frame, (640, 480))
    displ_frame = frame.copy()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    faces = detector.detect_faces(frame)
    if len(faces) > 0:     
        x,y,w,h = faces[0]['box']
        faceFrame =  frame[y:y+h, x:x+w]
        faceFrameResized = cv.resize(faceFrame, (160,160))
        test_frame = Get_Embedding(faceFrameResized)
        test_frame = [test_frame]
        ypreds = model.predict(test_frame)
        person = encodder.inverse_transform(ypreds)

        cv.rectangle(displ_frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv.putText(displ_frame, str(person[0]), (x,y-10), cv.FONT_HERSHEY_COMPLEX, 1.1, (0,255,0), 2)

    cv.imshow("FACR RECOGNITION WINDOW", displ_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Quiting...")
        break

cap.release()
cv.destroyAllWindows()

   