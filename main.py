import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


# Loading the dog breed model
model = load_model("dog_breed.h5")
class_names = ['Scottish Deerhound','Maltese Dog','Afghan Hound']

# Creating the app 
st.title("Dog Breed Prediction")
st.markdown("Upload the image of the dog : ")

# Uploading the dog image
dog = st.file_uploader("Choose an image : ",type=["png","jpeg","jpg"])
submit = st.button("Predict the breed of dog for the image")


if submit:
    if dog is not None:
        # Converting the image to byte code
        file_by = np.asarray(bytearray(dog.read()),dtype=np.uint8)
        # Opening the image
        open_cv = cv2.imdecode(file_by,1)
        
        # Display the image uploaded
        st.image(open_cv,channels="BGR")
        # Resizing the image
        open_cv = cv2.resize(open_cv,(224,224))
        # Convert the image into 4 dimension 
        open_cv.shape = (1,224,224,3)
        Y_pred = model.predict(open_cv)
        st.title("The dog breed of the uploaded image is : "+class_names[np.argmax(Y_pred)])


