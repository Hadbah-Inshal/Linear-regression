# Linear Regression on 2 Datasets (Gradio + Flask)

1. `pip install -r requirements.txt`
2. Put your 2 CSVs in `data/` and edit `config.py` (file name + target column).
3. `python train.py`        -> trains both models, saves metrics + plots
4. `python app_gradio.py`   -> http://127.0.0.1:7860
5. `python app_flask.py`    -> http://127.0.0.1:5000
