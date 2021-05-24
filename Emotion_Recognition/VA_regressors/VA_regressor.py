import vgg16_model_true
import vgg16_model
import vgg13_model
import cv2 as cv
from plotter import Plotter
import numpy as np
import tensorflow as tf

print("version",tf.__version__)

#### Select the model to use here.
USE_VGG16_OLD = False
USE_VGG16_NEW = True
USE_VGG13 = False
PATH = "models/"

if USE_VGG16_OLD:
    model=vgg16_model.get_model()
    # Load the previously saved weights
    model.load_weights(PATH+'model_custom_vgg_affectnet.h5')
    DIMENSION=(224,224)
elif USE_VGG16_NEW:
    model=vgg16_model_true.get_model()
    # Load the previously saved weights
    model.load_weights(PATH+'model_vgg16t_ccc_.h5')
    DIMENSION=(224,224)
elif USE_VGG13:
    model=vgg13_model.get_model()
    # Load the previously saved weights
    model.load_weights(PATH+'vgg13t_affectnet1.h5')
    DIMENSION=(64,64)


def extract_face(frame ):
    # extracting face and body and outputs tensor # add size of output image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 6)
    # extract ROIS
    if type(faces) is tuple:
        return False,False,False
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        ##simple scalings...
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        ##simple scalings...
        start = int(y - 0.25 * h)
        if start < 0:
            start = 0
        leftbottom=int(y +1.25*h)
        if leftbottom>720:
            leftbottom=720
        topright=int(x-w*0.25)
        if topright<0:
            topright=0
        bottom_right=int(x +1.25*w)
        if bottom_right>1280:
            bottom_right=1280
        roi = frame[start:leftbottom, topright:bottom_right]
    dim = DIMENSION
    resized = cv.resize(roi, dim, interpolation=cv.INTER_AREA)
    # print(resized.shape)
    # cv.imshow("frame",resized)
    return resized, (topright,start),(bottom_right,leftbottom)

def perform_prediction(frame):
    predicted=model.predict(frame[None,...]/255)
    return predicted*2-1

def concatenate_image(img1,img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # create empty matrix
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

    # combine 2 images
    vis[:h1, :w1, :3] = img1
    vis[:h2, w1:w1 + w2, :3] = img2
    return vis



cap=cv.VideoCapture(0)

dim_img = (480, 480)

# Create a plotter class object
p = Plotter(dim_img)

#fourcc = cv.VideoWriter_fourcc(*'MP4V')
#out = cv.VideoWriter('output.mp4', fourcc, 10.0, (int(480*2),480))

prediction_arr=[]
prev_valence=False
prev_arousal=False
valence =0
arousal=0
valence_array=[]
arousal_array=[]

N=10
while True:
    ret,frame=cap.read()
    if frame is None:
        break
    #extracting the frame
    face,top,bottom=extract_face(frame)
    if type(face) is not bool:
        frame=cv.rectangle(frame, top,bottom, (0, 0, 0), 2)
        #prediction
        prediction=perform_prediction(face)
        valence_array.append(prediction[0][0])
        arousal_array.append(prediction[0][1])
        if len(valence_array)>=10:
            valence=np.convolve(valence_array, np.ones(N) / N, mode='valid')[0]
            arousal=np.convolve(arousal_array, np.ones(N) / N, mode='valid')[0]
            valence_array=valence_array[1:]
            arousal_array=arousal_array[1:]

        #plotting prediction
        print(valence)
        coordinate_system=p.plot((valence,arousal))

        resized = cv.resize(frame, dim_img, interpolation=cv.INTER_AREA)
        vis=concatenate_image(resized,coordinate_system)
        cv.imshow("window",vis)
        #out.write(vis)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
#out.release()