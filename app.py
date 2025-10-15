# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
#
# # Add AVIF support for Pillow
# try:
#     import pillow_avif  # noqa: F401
# except ImportError:
#     st.warning("Install pillow-avif-plugin to enable AVIF support: pip install pillow-avif-plugin")
#
# # Load model (cached)
# @st.cache_resource
# def load_model():
#     model_path = r"C:\Users\favia\PycharmProjects\Chris_portfolio_multiproject\image_cnn\CATS_VS_DOGS\reports\models\resnet50_ft_aug_weights.keras"
#     model = tf.keras.models.load_model(model_path)
#     return model
#
# model = load_model()
#
# # Preprocessing function
# def preprocess_image(image: Image.Image) -> np.ndarray:
#     img = image.convert("RGB")
#     img = img.resize((224, 224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
#     return img_array
#
# # UI
# st.title("PetVision AI")
# st.write("Upload an image of a cat or dog (JPEG, PNG, WEBP, AVIF) and let the model predict.")
#
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp", "avif"])
#
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#
#     # Prediction with spinner
#     with st.spinner("Classifying..."):
#         img_array = preprocess_image(image)
#         preds = model.predict(img_array)
#         prob = tf.nn.softmax(preds[0]).numpy()
#
#     class_names = ["Cat", "Dog"]
#     predicted_class = class_names[np.argmax(prob)]
#     confidence = np.max(prob) * 100
#
#     st.subheader("Prediction")
#     st.success(f"**{predicted_class}** with {confidence:.2f}% confidence.")
#
#     st.write("### Class Probabilities")
#     for name, p in zip(class_names, prob):
#         st.progress(float(p))
#         st.write(f"- {name}: {p*100:.2f}%")


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Optional: add AVIF support for Pillow
try:
    import pillow_avif  # noqa: F401
except ImportError:
    st.warning("Install pillow-avif-plugin for AVIF support: pip install pillow-avif-plugin")

# Page config
st.set_page_config(
    page_title="PetVision AI 🐾",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load model (cached)
@st.cache_resource
def load_model():
    model_path = r"/02_image_cnn\CATS_VS_DOGS\reports\models\resnet50_ft_aug_weights.keras"
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Preprocessing function
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

# Hero banner
st.markdown(
    """
    <div style="text-align:center; padding: 10px; border-radius: 10px;">
        <h1 style="color:#FF9800;">🐾 PetVision AI</h1>
        <h3 style="color:#9E9E9E;">Cat vs Dog Classifier with ResNet50</h3>
        <p style="color:#BDBDBD;">Upload your pet photo and let AI decide 🐱🐶</p>
    </div>
    """,
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp", "avif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Layout: image on left, prediction on right
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        # Prediction with spinner
        with st.spinner("Classifying..."):
            img_array = preprocess_image(image)
            preds = model.predict(img_array)
            prob = tf.nn.softmax(preds[0]).numpy()

        class_names = ["Cat", "Dog"]
        predicted_class = class_names[np.argmax(prob)]
        confidence = np.max(prob) * 100

        st.subheader("Prediction")
        st.success(f"**{predicted_class}** with {confidence:.2f}% confidence.")

        st.write("### Class Probabilities")
        for name, p in zip(class_names, prob):
            st.progress(float(p))
            st.write(f"- {name}: {p*100:.2f}%")

        # Fun feedback
        if predicted_class == "Cat":
            st.balloons()
            st.write("😺 Looks like a cat!")
        else:
            st.snow()
            st.write("🐶 Woof! That's a dog!")
