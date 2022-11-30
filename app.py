from tensorflow.keras.models import load_model, Sequential
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pickle
import tensorflow as tf
import cv2
import re, string


def standardize(s):
  s = tf.strings.lower(s)
  s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
  s = tf.strings.join(['[START]', s, '[END]'], separator=' ')
  return s

from model import model

# Specify canvas parameters in application
# drawing_mode = st.sidebar.selectbox(
#     "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
# )


def drawHereMode():

    drawing_mode = "freedraw"
    st.header("Quick Draw Recognizer")
    st.subheader("Draw Here")

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider(
            "Point display radius: ", 1, 25, 3)
    # stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    # bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    #bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    bg_color = "#fff"
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        background_color=bg_color,
        #background_image=Image.open(bg_image) if bg_image else None,
        background_image= None,
        update_streamlit=realtime_update,
        height=250,
        width=250,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

    classes = ['alarm clock', 'angel', 'apple', 'backpack', 'banana', 'basket',
            'bed', 'bell', 'bicycle', 'binoculars', 'book',
            'bus', 'butterfly', 'camera', 'car (sedan)', 'cat',
            'chair', 'cloud', 'cow', 'cup', 'dog',
            'door', 'duck', 'ear', 'eye', 'eyeglasses',
            'fish', 'flying bird', 'guitar', 'hand', 'hat',
            'helicopter', 'horse', 'house', 'key', 'knife',
            'ladder', 'laptop', 'monkey', 'moon', 'pen',
            'person', 'radio', 'scissors', 'screwdriver', 'shoe',
            'socks', 'table', 'teacup', 'telephone']

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        # st.image(canvas_result.image_data)
        # pik_file = open("model_cnn.pkl","rb")
        # #cnn = pickle.load(pik_file)
        # cnn = tf.keras.models.load_model(pik_file)

        model = load_model('cnn_model.h5')
        img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2RGB)
        X = np.expand_dims(img, axis=0)
        images = np.vstack([X])
        # print(model.predict(images))
        val = model.predict(images)
        index = np.argmax(val[0])
        #print(val)
        print(index)
        print(classes[index])
        st.subheader("Predictions")
        st.markdown(classes[index])

    # if canvas_result.json_data is not None:
    #     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    #     for col in objects.select_dtypes(include=['object']).columns:
    #         objects[col] = objects[col].astype("str")
    #     st.dataframe(objects)

def uploadImageMode():
    st.header("Upload Image Here")
    try:
        image = st.file_uploader("Upload PNG File",type=["png","jpg"],key=1)
        show_file = st.empty()
        show_file.image(image)
        content = image.getvalue()
        original_img = cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
        original_img = cv2.resize(original_img,(250,250))
        classes = ['alarm clock', 'angel', 'apple', 'backpack', 'banana', 'basket',
                'bed', 'bell', 'bicycle', 'binoculars', 'book',
                'bus', 'butterfly', 'camera', 'car (sedan)', 'cat',
                'chair', 'cloud', 'cow', 'cup', 'dog',
                'door', 'duck', 'ear', 'eye', 'eyeglasses',
                'fish', 'flying bird', 'guitar', 'hand', 'hat',
                'helicopter', 'horse', 'house', 'key', 'knife',
                'ladder', 'laptop', 'monkey', 'moon', 'pen',
                'person', 'radio', 'scissors', 'screwdriver', 'shoe',
                'socks', 'table', 'teacup', 'telephone']
        model = load_model('cnn_model.h5')
        img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)
        X = np.expand_dims(img, axis=0)
        images = np.vstack([X])
        # print(model.predict(images))
        val = model.predict(images)
        index = np.argmax(val[0])
        #print(val)
        print(index)
        print(classes[index])
        st.subheader("Predictions")
        st.markdown(classes[index])
    except AttributeError:
        print("AttributeError Found")




def captionGenerationMode():
    st.header("Caption Generator")
    st.header("Upload Image Here")
    try:
        image = st.file_uploader("Upload PNG File",type=["png","jpg"],key=2)
        show_file = st.empty()
        show_file.image(image)
        content = image.getvalue()
        original_img = cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
        original_img = cv2.resize(original_img,(224,224))
        original_img = original_img.reshape(224,224,3)
        k = model.simple_gen(original_img)
        print(k)
        # print("ok")
        # print(dir(model))
        # model = Sequential()
        # model.built = True
        # model.load_weights('caption_model.h5')
        # # img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)
        # X = np.expand_dims(original_img, axis=0)
        # images = np.vstack([X])
        # print(dir(model))
        # print(model.predict(images))
    except AttributeError:
        print("AttributeError Found")





tabs = st.tabs(('Draw Here','Upload Image','Caption Generator'))

with tabs[0]:
    drawHereMode()
with tabs[1]:
    uploadImageMode()
with tabs[2]:
    captionGenerationMode()