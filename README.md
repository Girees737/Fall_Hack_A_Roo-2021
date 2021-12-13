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
       probobilities of 7 emtions in the final layer.<br>
       
#### Data Source 
2. Dataset used is fer2013 to train the mini_xception state of art model which was downloaded from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data. It contains facial images with seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

3. Imported the required libraries to build a MINI_XCEPTION model as shown below.<br>
 ![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Documentation/1.PNG)<br>

#### Data Loading
4. Defined a method called load_fer2013 method as shown below, it reads the csv file and convert pixel sequence of each row in image of dimension 48 * 48 and returns the faces and emotions of the image.<br>
 ![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Documentation/2.PNG)<br>

#### Data Preprocessing
 5. defined a mothod called preprocess_input to pre-process images by scaling them between -1 to 1. Images is scaled to [0,1] by dividing it by 255. Further, subtraction by 0.5 and multiplication by 2 changes the range to [-1,1] as it was observed that [-1,1] has been found a better range for neural network models in computer vision problems.<br>
 
  ![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Documentation/3.PNG)<br>

#### CNN model : Mini Xception
6. Initialized the model training parameters as shown below and data generator for data augumentation during training process.<br>

 ![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Documentation/4.PNG)<br>
 
7. Initiated the kerner regularizer as L2Regularization to avoid overfitting and built the mini_Xception architecture using Functional API model shown below.
8. Added the callback parameters like model_checkpoint, early stopping and ReduceLRonPlateu.<br>

 ![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Documentation/5.PNG)
 ![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Documentation/6.PNG)
 ![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Documentation/7.PNG)<br>
 
 9. We could also see the keras visual representation of the model as follows.

![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Documentation/10.PNG)<br>


#### Model Training:

10. Trained the model by calling fit_generator method, its was designed to train untill 110 epochs with the Early stopping, model check point and ReduceLRonPlateau as callbacks.

 ![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Documentation/8.PNG)<br>

11. Observed that the model stopped learning and updating weights after epoch 82 and reached the train accuracy of 0.69 and validation accuracy of 0.65 as shown below.
12. Instance of that model was saved to a path with the name below with the help of model check point.

![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Documentation/9.PNG)<br>


#### Model Evaluation:

13. Evaluated the model with test data and observed the performance is same as we have seen at epoch 82 during training.

![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Documentation/11.PNG)<br>

14. Plotted the performce of the model on test and train data with respect to accuracy and loss during training.

![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Screenshots/Training_Val_Accuracy.PNG)<br>

### Testing the model in Real time:

15. Tested the saved model on an image with multiple faces, It has identidied all the faces and assigned an emotion to them as shown below.

![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Screenshots/predictions%20on%20Multiple_Faces.PNG)<br>

### Testing the model on a live stream video:

16. Tested the model on web cam stream and projected the probobilities of emotions on a given frame of image in the video as shown below.

![](https://github.com/Girees737/Fall_Hack_A_Roo-2021/blob/main/Screenshots/V_Happy.PNG)<br>



