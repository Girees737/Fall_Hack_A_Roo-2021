# Facial Emotion Recognition

## Team Name: Hack Steet Boys
1. Gireesh Kumar Muppalla
2. Ajay Kumar Kancharla
3. Mohan Reddy Tamma
4. Vishnumonish Kankanala

Video Link: (if not submitting in class) : https://youtu.be/8CVzClMeuAM

### Project Name : Facial Emotion Recognition

### Aim:
The project Facial Emotion Recognition aims to recognize the emotions of a person like happy, sad, anger, surprise, scared, neutral and disguse. 

### Motivation:
The modern life style has come up with modern problems and stress, it is said true that we have gone through at least one type of mental issues in our life time, it is really helpful to have someone who understands what we go through by looking at our emotional state and gestures like our facial expression.  This projects solves the exact same problem by providing near to accurate solution over detecting the above mentioned types of emotions.

### Applications and Feature Scope: 
This FER solution is useful in real time applications like employee welfare, medical field, psychometric examinations etc.

### Technology : 
Deep Learning

### Programming Language and IDE: 
Python 3.6 , Jupyter Notebook

### Libraries:
1. Tensorflow
2. Keras
3. Sklearn
4. Matplotlib
5. Numpy
6. Pandas
7. OpenCV

### Implementation:

1. It comprises of two step process i.e. face detection (bounded face) in an image followed by emotion detection on the detected bounded face. <br>
    a. Haar feature-based cascade classifiers : It detects frontal face in an image well. It is real time and faster in comparison to other face detector. This blog-post uses an        implementation from Open-CV.<br>
    b. Xception CNN Model (Mini_Xception, 2017) : We will train a classification CNN model architecture which takes bounded face (48 * 48 pixels) as input and predict       
       probobilities of 7 emtions in the final layer.

