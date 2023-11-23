from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16


def load_model():
    model = VGG16(weights="imagenet",
                  include_top=False,
                  input_shape=(224,224,3),
                  pooling='avg')
    return model


def predict_frame(model, frame):
    model = load_model()
    preprocessed_frame = preprocess_input(frame)
    prediction = model.predict(preprocessed_frame)
    return prediction
