from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import cv2
import numpy as np
import sys

# parameters for loading data and images
detection_model_path = 'C:\\Users\\muppa\\Downloads\\haarcascade_frontalface_default.xml'
emotion_model_path = 'C:\\Users\\muppa\\Downloads\\model_mini_XCEPTION_epoch02d_val_acc_2f.hdf5'
# img_path = sys.argv[1]
# img_path = 'C:\\Users\\muppa\\Downloads\\FER\\test\\happy\\im1021.jpg'
img_path = 'C:\\Users\\muppa\\OneDrive\\Desktop\\einstin_batch1.jpg'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def predict(img_path):
    # reading the frame
    orig_frame = cv2.imread(img_path)
    frame = cv2.imread(img_path, 0)
    faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    # faces1 = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)


    if len(faces) > 0:
        for i in faces:
            print(i)
            print(type(i))
            #         faces = sorted(i, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = i
            roi = frame[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    # cv2.imshow('test_face', orig_frame)
    import os
    os.chdir("C:\\Users\\muppa\\PycharmProjects\\Facial_Recognition\\static")
    # os.getcwd()='C:\\Users\\muppa\\PycharmProjects\\Facial_Recognition\\Images'
    cv2.imwrite('C:\\Users\\muppa\\PycharmProjects\\Facial_Recognition\\Static\\predicted_Image.jpg', orig_frame)
    return orig_frame

# if (cv2.waitKey(20000) & 0xFF == ord('q')):
#     sys.exit("Thanks")
# cv2.destroyAllWindows()