import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2 as cv
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet



class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def Extract_Face(self,filename):
        img = cv.imread(filename)
        img =cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x), abs(y)
        img_face = img[y:y+h, x:x+w]
        face_arr = cv.resize(img_face, (self.target_size))
        return face_arr
    
    def load_faces(self,dir):
        FACES = []

        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                single_face = self.Extract_Face(path)
                FACES.append(single_face)
            except Exception as e:
                print(f"[SKIPPED] {path} -> {e}")
        return FACES
    
    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory +'/'+ sub_dir + '/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Laoded Successfully {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)
        return np.asarray(self.X) , np.asarray(self.Y)
    


faceloading = FACELOADING(" ___Path/of/faces/directory to train them___")    # gave path where you store faces_folders to train
X,Y = faceloading.load_classes()


# face_net ----------------
embedder = FaceNet()

def Get_Embedding(face_img):
    face_img = face_img.astype('float32')  #3D (160x160x3)
    face_img = np.expand_dims(face_img, axis=0)  #4D (Nonex160x160x3)
    yhat = embedder.embeddings(face_img)
    return yhat[0]  

EMBEDDED_X = []
for img in X:
    EMBEDDED_X.append(Get_Embedding(img))
EMBEDDED_X = np.asarray(EMBEDDED_X)

np.savez_compressed("Faces_Embeddings.npz", EMBEDDED_X,Y)


# SVM MODEL----------------
from sklearn.preprocessing import LabelEncoder 
encoder = LabelEncoder()
encoder.fit(Y)
Y= encoder.transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X,Y, shuffle= True, random_state=17)


from sklearn.svm import SVC
model = SVC(kernel='linear',probability= True)
model.fit(X_train, Y_train)

ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_train, ypreds_train)
accuracy_score(Y_test, ypreds_test)





# ====================================================
import pickle

with open("face_artifacts.pkl", "wb") as f:
    pickle.dump(
        {
            "svm": model,
            "encoder": encoder
        },
        f
    )





















#-------------------------know recognization part-----------------------------

# This section is to check weather face_train is working properly or not by giving imput image t_im and get dezire result


"""detector = MTCNN()
t_im = cv.imread(" ___Gave/Image/Path ___ ")  # provide image path with image name (e.g: /home/pictures/my_pic/unknown.jpg)
t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
x,y,w,h = detector.detect_faces(t_im)[0]['box']

t_im = t_im[y:y+h, x:x+w]
t_im = cv.resize(t_im, (160,160))
test_im = Get_Embedding(t_im)

test_im =[test_im]
ypreds= model.predict(test_im)

print(f"name value_index: {ypreds}")
person = encoder.inverse_transform(ypreds)
print(f"Identified Person: {person}")
"""

