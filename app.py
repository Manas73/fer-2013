import streamlit as st
import cv2
from PIL import Image
import numpy as np
from model import FacialExpressionModel
import pandas as pd
import plotly.graph_objs as go
import time


@st.cache
def load_image(img):
    im = Image.open(img)
    return im


facec = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised",
]


def get_frame(our_image):
    model = FacialExpressionModel("model.json", "model_weights.h5")
    image = np.array(our_image.convert("RGB"))
    new_image = cv2.cvtColor(image, 1)
    gray_fr = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)

    for (x, y, w, h) in faces:
        fc = gray_fr[y : y + h, x : x + w]

        roi = cv2.resize(fc, (48, 48))
        pred, probs = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

        cv2.putText(new_image, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(new_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # _, jpeg = cv2.imencode('.jpg', fr)
    return new_image, faces, probs


def main():
    """Facial Expression Detection"""

    st.title("Facial Expression Detection App")
    st.subheader("Facial Expression Detector")
    st.write(
        """
    ## **Inspiration** \n
    Goodfellow, I.J., et.al. (2013). Challenged in representation learning: A report of three machine learning contests. Neural Networks, 64, 59-63. doi:10.1016/j.neunet.2014.09.005

    This project used the FER-2013 dataset which consisted of 48x48 pixel grayscale images of faces. The model will classify each facial expression into one of seven categories: \n
    1. Angry \n
    2. Disgust \n
    3. Fear \n
    4. Happy \n
    5. Sad \n
    6. Surprise \n
    7. Neutral \n
    The training set consists of 28,709 examples and the public test set consists of 3,589 examples.
    """
    )
    activities = [
        "Detect Facial Expression",
    ]
    choice = st.sidebar.selectbox("Select an activity", activities)
    if choice == activities[0]:
        image_file = st.file_uploader(
            "Upload Image", type=["jpg", "png", "jpeg"]
        )
        if image_file is not None:
            our_image = Image.open(image_file)
            placeholder = st.empty()
            placeholder.image(our_image)

        if st.button("Process"):
            result_img, result_faces, probs = get_frame(our_image)
            placeholder.image(result_img)
            probability = pd.DataFrame(probs, columns=emotions).transpose()
            # st.write(probability)
            fig = go.Figure([go.Bar(x=emotions, y=probability[0])])
            fig.update_layout(
                title={
                    "text": "Probability of each Expression",
                    "y": 0.9,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                },
                xaxis_title="Expression",
                yaxis_title="Probability",
            )
            st.write(fig)

    st.write(
        """
    ## **Methodology** \n
    The model was built using Tensorflow for the classification of facial expressions. Once an image of a person is uploaded or input from the webcam is passed, the faces are detected using OpenCV, and the application then classifies the facial expression.

    ## **Results** \n
    The Facial Expression classifier has an accuracy of 65.1% which among the best for this particular dataset. For context, the human classification accuracy for the dataset was 65 ± 5%.

    """
    )


if __name__ == "__main__":
    main()
