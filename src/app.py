import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request
import model.prediction_model as yapayZeka


app = Flask(__name__)

@app.route("/" , methods=["GET", "POST"])
def index():
    if request.method == "GET":
            return render_template("index.html")
    else:
        text = request.form["news-area"]
        if not text.strip():
            return render_template("index.html")
        else:
            if len(text.split()) > 4:
                pred,prob = yapayZeka.predict(text)
                print(pred,"-",prob)
                return render_template("index.html",pred=pred,prob=str(prob))
            else:
                return render_template("index.html",error="Yetersiz uzunlukta haber girdiniz.")

        


if __name__ == "__main__":
    app.run(debug=False)