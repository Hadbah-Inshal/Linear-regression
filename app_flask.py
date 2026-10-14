"""Flask UI.  Run:  python app_flask.py   ->  http://127.0.0.1:5000"""
from flask import Flask, abort, render_template, request

from common import DATASETS, load_meta, predict

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", datasets=DATASETS)


@app.route("/<key>", methods=["GET", "POST"])
def model_page(key):
    if key not in DATASETS:
        abort(404)
    meta = load_meta(key)
    result, error, values = None, None, {}
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            result = f"{predict(key, values):,.2f}"
        except (ValueError, KeyError):
            error = "Please fill all fields with valid values."
    return render_template("predict.html", key=key, meta=meta, datasets=DATASETS,
                           result=result, error=error, values=values)


if __name__ == "__main__":
    app.run(debug=True)
