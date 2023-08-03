# Iluminado (Data Science Capstone Project)

## Deep Ensemble Learning for Retinal Image Classification (CNN)

## Problem Statement

The World Health Organization (2019) estimates there are 2.2 billion vision-impaired people worldwide, of whom at least 1 billion could have been prevented or are still untreated. As far as eye care is concerned, the world faces many challenges, including inequalities in coverage and quality of prevention, treatment, and rehabilitation services. There is a shortage of trained eye care providers and the poor integration of eye care services into major health systems. It is my goal to galvanize action to address these challenges as part of my data science capstone project — Iluminado.

## Project Aim
In view of the aforementioned issue, my project aims to train a deep learning ensemble model that is highly accessible for low-income families, which can perform an initial disease risk diagnosis at a low-cost. By using my model, ophthalmologist can determine whether an intervention is needed immediately based on the prediction from the model and retinal image itself.

My motivations for this project are two-folded:

* I personally had undergone ICL surgery (a.k.a Implantable Collamer Lens Surgery). After witnessing how AI and machines play a major role in the diagnostic process, I decided to utilize what I have learned in data science to create a deep learning classifier model to help ophthalmologist land on a better informed diagnosis accordingly.

* As an aspiring data scientist, my ultimate aim to create a positive impact - no matter how small it is - to the world. This capstone project fulfills my little passion project and bring that long-envisioned prototype to reality.


## Table of Contents
1. [Source of the Dataset](#Source-of-the-Dataset)
2. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
3. [Preprocessing](#Preprocessing)
    * [Data Cleaning & Sorting](#Data-Cleaning-&-Sorting)
    * [Flip and Rescale All Images](#Flip-and-rescale-all-images)
4. [Neural Network Architecture](#neural-network-architecture)
5. [Model Evaluation](#Model-Evaluation)
6. [Deployment on Streamlit](#Deployment-on-Streamlit)
7. [Future Adaptations / Improvements](#Future-Adaptations-/-Improvements)
8. [References](#References)
9. [Medium Blog Post](#Medium-Blog-Post)

## Source of the Dataset

The dataset could be found in here: https://www.mdpi.com/2306-5729/6/2/14

Caution: The datasets are not provided in Github due to its large file size

There is a public available image dataset called the Retinal Fundus Multi-Disease Image Dataset (RFMiD), provided by OphthAI, which comprises 3200 fundus images that were captured with three different fundus cameras and annotated by two senior retinal experts based on adjudicated consensus.

The images were extracted from the thousands of examinations done during the period 2009–2020. Both high-quality and low-quality images are selected to make the dataset challenging.

The dataset has been divided into 3 parts, consisting of a training set (60% or 1920 images), an evaluation set (20% or 640 images), and a test set (20% or 640 images).On average, the disease-wise ratio in the training, evaluation and testing sets are 60 ± 7 %, 20 ± 7%, and 20 ± 5%, respectively. The fundamental purpose of this dataset is to tackle a broad array of eye illnesses seen in daily clinical practice. There are 45 categories of diseases/pathologies identified in total. These labels can be found in the three CSV files; namely, the RFMiD_Training_Labels.CSV, the RFMiD_Validation_Labels.CSV, and the RFMiD_Testing_Labels.CSV.



### Where does the image come from?

This image was captured using a tool known as the fundus camera — a specialized low power microscope attached to a flash-enabled camera to photograph the fundus, a retinal layer at the back of the eye.
Most fundus cameras today are handheld, so patients need only look straight into the lens. A bright flash indicates that a photograph of the fundus has been taken.

![Fundus Camera Specification](http://localhost:8892/lab/tree/GA/project-capstone%20(Illuminado%20Capstone%20Project)/Streamlit%20Deployment%20Example.jpg)

Handheld cameras are beneficial since they can be carried to different locations and can accommodate patients who have unique needs, such as wheelchair users. In addition, any employee who has received the required training can operate the camera, allowing underserved diabetics to have their annual exams quickly, safely, and efficiently.

(Reference: Fundus Retinal Imaging Systems)


## Exploratory Data Analysis

![Specifications table](http://localhost:8805/files/GA/project-capstone%20(Illuminado%20Capstone%20Project)/Specification%20Tables/Specifications%20table.png?_xsrf=2%7C4d329aca%7C8c7d48209f868bf0914a034eaa13e7d9%7C1673664406)

![Dataset Description](http://localhost:8805/lab/tree/GA/project-capstone%20(Illuminado%20Capstone%20Project)/Specification%20Tables/Dataset%20Description.png)

![Pathologies Type Count](http://localhost:8805/lab/tree/GA/project-capstone%20(Illuminado%20Capstone%20Project)/Specification%20Tables/Pathologies%20Type%20Count.png)

## Data Cleaning & Sorting
To sort the images properly, I used the CSV file as a reference to filter the (disease_risk = 1) and (disease_risk = 0) images accordingly. Then, use the import the OS system to help sort the images into the designated file 'yes_disease' and 'no_disease' accordingly. The same goes to multi-classification process.


## Preprocessing

The preprocessing pipeline is the following:

```
train_augmented = ImageDataGenerator(rescale = 1/255.,
                                     shear_range= 0.1,
                                     zoom_range= 0.2,
                                     horizontal_flip = True,
                                     vertical_flip = True)
```


### Download All Images to my own computer
The images were downloaded via OphthAI. Loading those images on Jupyter lab took me nearly 1 hour. Not to mention the CNN models - all of them consumed a lot of computing power, as well as time. All images are then allocated to in their respective folders according to disease risk and disease type, and expanded from their compressed files. In total, there are more than 40GB occupied by these images.

### Crop and Resize All Images
All images were scaled down to 150 by 150 (which is a compromise for less computing power) Despite taking less time to train, the detail present in photos of this size is about 1424 by 2144.

Do note that, both high-quality and low-quality images are selected to make the dataset challenging. Some images were taken in a different lighting environment, which might possibly affect the modeling result.

### Flip and Rescale All Images
All images were horizontally and vertically flipped.


## Neural Network Architecture

Keras is used to build the model, while TensorFlow is used for the backend.

For predicting whether the image is classified as 'yes_disease' or 'no_disease', Iluminado utilized several pre-trained convolutional base (InceptionV3,  Xception, VGG16, MobileNetV2, EfficientNetB5 and SE-ResNeXt) + a top layer that has a structure listed as follows:

```
top_layer = Sequential()
top_layer.add(Dense(100, activation = 'relu'))
top_layer.add(Dropout(0.2))
top_layer.add(Flatten())
top_layer.add(Dense(512,activation="relu"))
top_layer.add(Dense(1, activation = 'sigmoid'))

```


First, it has a dense layer of 100 activated with 'relu', then a dropout later. Next, the image is flattened then pass thorugh another dense layer of 512, and finally to the output layer, consisting of 1 sigmoid output layer.

The same goes to multi-classification of Diabetic retinopathy (DR), Media Haze (MH) and Optic disc cupping (ODC) respectively. The only difference is the output layer, where I need to change to 3 classes activated by 'softmax':


```
top_layer.add(Dense(3, activation = 'softmax'))

```

The reason why those are the target classes to predict is because they are the top 3 categories that have the most photos to support the modeling process.



## Model Evaluation
Iluminado was my ideal prototype to show the percentage of disease risk to patient, so that they were better informed with their eye condition, as well as facilitate the diagnostic process of the ophthalmologist. The following table highlights the performances of each pre-trained model with different epochs.

Binary Classification:
| Architectures | Epochs | Training loss | Testing loss | Training accuracy | Testing accuracy |
|---|---|---|---|---|---|
| InceptionV3 | 70 | 0.01 | 2.11 | 0.99 | 0.74 |
| Xception | 70 | 0.00 | 2.22 | 0.99 | 0.73 |
| VGG16 | 30 | 0.17 | 0.98 | 0.92 | 0.73 |
| MobileNetV2 | 30 | 1.78 | 2.98 | 0.98 | 0.73 |
| EfficientNetB5 | 20 | 0.5 | 0.52 | 0.79 | 0.74 |
| SE-ResNeXt | 20 | 0.51 | 0.51 | 0.79 | 0.78 |


Multi Classification:
| Architectures | Epochs | Training loss | Testing loss | Training accuracy | Testing accuracy |
|---|---|---|---|---|---|
| InceptionV3 | 50 | 0.54 | 3.81 | 0.84 | 0.41 |
| Xception | 50 | 1.00 | 1.02 | 0.47 | 0.50 |
| VGG16 | 20 | 0.03 | 2.63 | 0.98 | 0.47 |
| MobileNetV2 | 20 | 0.06 | 2.98 | 0.97 | 0.45 |
| EfficientNetB5 | 30 | 0.12 | 4.40 | 0.95 | 0.47 |
| SE-ResNeXt | 20 | 0.07 | 3.92 | 0.96 | 0.41 |


Training accuracy is still high, but testing accuracy was a bit lower than expected. 

This may be because of an imbalanced dataset, there are 1519 (yes_disease) images but only 401 (no_disease) images. Therefore, the model was trained to predict disease images more than healthy images. One positive thing about this is that it could lower false negatives. As people would rather have false alarms than delayed treatments to cure their eyes. (Even though false alarms might not be pleasant as well). The second reason for low classification accuracy is because of overlapping disease types and a small dataset. If more images can be fed into the model, then it will certainly boost the accuracy and balance both sides.

## Deployment on Streamlit


![Deployment Example](http://localhost:8892/lab/tree/GA/project-capstone%20(Illuminado%20Capstone%20Project)/Streamlit%20Deployment%20Example.jpg)

To fulfill my wishes of turning a prototype into a reality, I decided to deploy the model uing streamlit. 
After the user has dragged an image to the platform, it will automatically process the image through the model, and it will then display the disease risk percentage to inform users about their eye conditions.

Hopefully, one day, the model will become fully mature and can be turned into an online platform that is readily available for low-income families to check their eye condition without lining up at clinic and wait for weeks.


## Future Adaptations / Improvements
1. Besides disease risk prediction, it is hoped that it could also display the type(s) of eye disease that the image is classified to with the percentage shown. But then that requires a lot more data to train.

2. Apart from just processing one image, hopefully, several images could be processed at the same time with a list of patients' name and disease risk prediction generated. So that, the eye clinic / hospital can store and display those data more efficiently in case there is more than 1 patient at the time.

3. Online machine learning should be incorporated, where data is acquired sequentially and is utilized to update the best predictor for future data at each step. To put it back to my capstone context, the model should improves itself automatically whenever new data arrives in real-time. The learning algorithm’s parameters are updated after learning from each individual training instance. 

4. In addition to the diagnostic nature of the app, not only do the patient can choose to keep the diagnostic result, but they can also choose to allow the diagnosis to be automatically sent to the patient’s preferred ophthalmologist as a medical reference and stored in a stable storage system (based on the patient's will). So that the patient can be follow-up appropriately with sutiable measures according to different development phrases.

## References

1. [Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Detection Research](https://www.mdpi.com/2306-5729/6/2/14)

2. [Fundus Retinal Imaging Systems](https://retinalscreenings.com/blog/fundus-photography-and-the-importance-of-quality-retinal-images/)

3. [World Health Organization: Blindness and vision impairment](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment)

## Medium Blog Post

1. [Deep Ensemble Learning for Retinal Image Classification (CNN)](https://medium.com/@cathykam2006/deep-ensemble-learning-for-retinal-image-classification-cnn-8a8ce73870c0)

