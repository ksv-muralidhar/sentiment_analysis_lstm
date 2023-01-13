from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import cloudpickle
import pandas as pd
import numpy as np
from preprocessing.text_preprocessing import TextPreprocessor
from tensorflow.keras import models
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["SM_FRAMEWORK"] = "tf.keras"


app = FastAPI()
with open(os.path.join(".", "models", "preprocessors.bin"), "rb") as f:
    tokenizer, maxlen, padding, truncating = cloudpickle.load(f)

model = models.load_model(os.path.join(".", "models", "review_sentiment_model.h5"), compile=False)


class Data(BaseModel):
    """
    Data dictionary for data type validation
    """
    text: str


@app.post("/")
def prediction(data: Data):
    """
    Processes the API request and returns a prediction
    """
    try:
        txt_ser = pd.Series(data.dict())  # converting api data dict to df
        tp = TextPreprocessor()
        preprocessed_str = tp.preprocess(txt_ser, dataset="test")
        txt_seq = tokenizer.texts_to_sequences(preprocessed_str)
        txt_seq = pad_sequences(txt_seq, maxlen=maxlen, padding=padding, truncating=truncating)
        pred = model.predict(txt_seq)[0]
        confidence_level = [np.round(p * 100, 2) for p in pred]
        response = {"result": ["negative", "positive"], "confidence": confidence_level}
        json_compatible_item_data = jsonable_encoder(response)
        response = JSONResponse(content=json_compatible_item_data)

    except Exception as e:
        # executes in case of any exception
        response = {"result": str(e), "confidence": "-"}
        json_compatible_item_data = jsonable_encoder(response)
        response = JSONResponse(content=json_compatible_item_data)
    return response


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5001)
