import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import streamlit as st
import numpy as np

st.header('Image Classification Model')

# Load model
try:
    model = load_model('Image_classify.keras')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Categories
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
    'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
    'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

img_height = 180
img_width = 180

# Choose input method
input_method = st.radio('Select input method:', ['Text Input', 'Upload File'])
image = None
original_displayed = False

if input_method == 'Text Input':
    image_path = st.text_input('Enter image filename or path:', 'Apple.jpg')
    if image_path:
        # Show original full-resolution image
        try:
            st.subheader("Original Image")
            st.image(image_path, use_container_width=True)
            original_displayed = True
        except Exception:
            # ignore display errors for text paths
            pass

        # Load & resize for model with bicubic interpolation
        try:
            image = tf.keras.utils.load_img(
                image_path,
                target_size=(img_height, img_width),
                interpolation='bicubic'
            )
        except FileNotFoundError:
            st.error(f"File not found: {image_path}")
        except Exception as e:
            st.error(f"Error loading image: {e}")

elif input_method == 'Upload File':
    uploaded_file = st.file_uploader('Upload an image file:', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Show original full-resolution upload
        st.subheader("Original Image")
        st.image(uploaded_file, use_container_width=True)
        original_displayed = True

        # Load & resize for model with bicubic interpolation
        try:
            image = tf.keras.utils.load_img(
                uploaded_file,
                target_size=(img_height, img_width),
                interpolation='bicubic'
            )
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")

# Predict and display
if image is not None:
    # If original wasn't shown (e.g. text input display failed), show the resized version
    if not original_displayed:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    # Convert to array (0–255); model will rescale internally
    img_arr = tf.keras.utils.img_to_array(image)
    img_batch = tf.expand_dims(img_arr, 0)

    # Predict
    predict = model.predict(img_batch)
    score = tf.nn.softmax(predict[0])

    # Results
    pred_class = data_cat[np.argmax(score)]
    conf = np.max(score) * 100

    st.write(f"Prediction: {pred_class}")
    st.write(f"Confidence: {conf:.2f}%")
else:
    st.info('Please provide an image via the selected input method.')
