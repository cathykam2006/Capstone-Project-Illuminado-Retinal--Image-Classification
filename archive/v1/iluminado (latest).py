#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import os
from io import BytesIO
from tensorflow.keras.models import load_model
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


st.title("Iluminado")
st.header("Initial Eye Disease Risk Diagnosis")
st.subheader("This convolutional net aims to assist ophthalmologist to determine whether an intervention is needed immediately based on retinal fundus photography.")
st.subheader("You may upload your retinal image down below taken the fundus camera to see your result of an initial diagnosis.")

from PIL import Image
image = Image.open('Illuminado Cover Picture.png')

st.image(image, caption='Project Cover Picture')


agree = st.checkbox("Consent form: Your participation in this initial diagnosis is voluntary. After you tick the consent form, the image you submitted will be kept as part of data collection for advancing the current model.")

if agree:
    st.text("I have read and I understand the provided information. I understand that my participation is voluntary and I am willing to submit my retinal image for data collection purposes. I voluntarily agree to take part in this diagnosis.")

uploaded_file = st.file_uploader("Drag and drop your retinal image here", accept_multiple_files=False, type = ['png', 'jpeg', 'jpg'])

if not uploaded_file:
  st.warning('Please upload an image.')
  st.stop()
st.success('Thank you for uploading your image.')

if uploaded_file is not None: 
    user_file_path = os.path.join("users_uploads/", uploaded_file.name)
    with open(user_file_path, "wb") as user_file:
        user_file.write(uploaded_file.getbuffer())

    image_read = tf.keras.utils.load_img(user_file_path, target_size = (1424, 2144))
    image_read = tf.keras.utils.img_to_array(image_read)
    image_read = np.array([image_read])
    
    #train_augmented = ImageDataGenerator(rescale = 1/255.,
                                     #shear_range= 0.1,
                                     #zoom_range= 0.2,
                                     #horizontal_flip = True,
                                     #vertical_flip = True)

    #train_data_augmented = train_augmented.flow_from_directory('users_uploads/')


    #st.image(initialize_model(model_name, image))

    #image_data = Image.open(BytesIO(uploaded_file))
    st.write("filename:", uploaded_file.name)
    st.write(image_read)
    
conv_base = load_model('../GA/conv_base.keras')
top_layer = load_model('../GA/top_layer.keras')
model = load_model('../GA/model.keras')

#st.write(train_data_augmented)

#first_prediction = conv_base.predict(image_read)
#prediction = top_layer.predict(first_prediction)

prediction = model.predict(image_read)

#final_prediction = top_layer.predict(prediction)

st.text("If the result is or close to 1, it means there is a high-risk of eye disease.")
st.text("If the result is or close to 0, it means your eye is healthy and there is a low-risk of eye disease.")

st.write(f'The predicted image is {prediction[0]}')


