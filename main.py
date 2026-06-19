import os
import requests
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import io

app = FastAPI()

MODEL_PATH = "image_classification_mobilenetv2.keras"

MODEL_URL = "https://huggingface.co/Ghanshyam51/image_classification_mobilenetv2/resolve/main/image_classification_mobilenetv2.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    
    r = requests.get(MODEL_URL)

    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

    print("Model Downloaded Successfully")

model = load_model(
    MODEL_PATH,
    compile=False
)

CLASS_NAMES = [
    "Attire",
    "Decorationandsignage",
    "Food",
    "misc"
]

@app.get("/")
def home():
    return {
        "message": "Image Classification API Running Successfully"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    img = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    img = img.resize((128,128))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_class = CLASS_NAMES[
        np.argmax(prediction)
    ]

    confidence = float(
        np.max(prediction)
    )

    all_predictions = {
        CLASS_NAMES[i]: round(
            float(prediction[i]) * 100,
            2
        )
        for i in range(len(CLASS_NAMES))
    }

    return {
        "prediction": predicted_class,
        "confidence": round(
            confidence * 100,
            2
        ),
        "all_predictions": all_predictions
    }