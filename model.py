from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.15
# session = tf.compat.v1.Session(config=config)


class FacialExpressionModel(object):
    emotions = [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Neutral",
        "Sad",
        "Surprised",
    ]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as f:
            loaded_model_json = f.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return (
            FacialExpressionModel.emotions[np.argmax(self.preds)],
            self.preds,
        )
