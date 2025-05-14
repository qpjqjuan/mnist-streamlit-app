import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load your trained model
model = load_model("mnist_cnn_model.h5")

st.title("MNIST Digit Classifier")
st.write("Upload a 28x28 grayscale image of a handwritten digit to predict it.")

uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)                  # MNIST digits are white on black

    st.image(image, caption="Uploaded Image", use_column_width=False)

    # Resize and normalize
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.subheader(f"Predicted Digit: {predicted_digit}")
    st.bar_chart(prediction[0])
