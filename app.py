from flask import Flask, render_template, request
import requests
import numpy as np
from utils.get_api_url import get_api_url
from flask_cors import cross_origin, CORS


app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
API = get_api_url()  # get the URL of API. if no env var is present it defaults to localhost


@app.route("/")
@cross_origin()
def index():
    """
    displays the index.html page
    """
    return render_template("index.html")


@app.route("/", methods=["POST"])
@cross_origin()
def form_prediction():
    """
    Returns the API response based on the inputs filled in the form.
    """
    try:
        # assigning the inputs from the form to respective variables.
        text = request.form["text_input"]
        # input json to the API in the required format
        request_json = {"text": text}

        error = ""
        response = requests.post(url=API, json=request_json)
        if response.status_code == 200:
            response_json = response.json()
            predicted_label = response_json.get("result")
            confidence_level = np.array(response_json.get("confidence"))
            if confidence_level != "-":
                idx_max = np.argmax(confidence_level)
                prediction = predicted_label[idx_max]
                confidence = confidence_level[idx_max]
                font_col = "green"
                if prediction != "positive":
                    font_col = "red"
                result = f'<h3 style="color: {font_col};">{prediction.title()} ({confidence}%)</h2>'
            else:
                raise Exception(predicted_label)
        else:
            result = f'Error: Details at the bottom'
            error = response.json()
    except Exception as e:
        result = f'Error: Details at the bottom'
        error = e
    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
