"""Gradio UI.  Run:  python app_gradio.py   ->  http://127.0.0.1:7860"""
import gradio as gr

from common import DATASETS, load_meta, predict


def build_tab(key):
    meta = load_meta(key)
    names, inputs = [], []
    for f in meta["features"]:
        names.append(f["name"])
        if f["type"] == "num":
            inputs.append(gr.Number(label=f"{f['name']} (range {f['min']:g} to {f['max']:g})",
                                    value=f["default"]))
        else:
            inputs.append(gr.Dropdown(f["options"], value=f["options"][0], label=f["name"]))

    out = gr.Textbox(label=f"Predicted {meta['target']}")

    def run(*vals):
        return f"{predict(key, dict(zip(names, vals))):,.2f}"

    gr.Button("Predict", variant="primary").click(run, inputs, out)
    m = meta["metrics"]
    gr.Markdown(f"**Model performance (test set):** R² = {m['R2']} | MAE = {m['MAE']} | RMSE = {m['RMSE']}")
    gr.Image(f"static/{key}.png", label="Actual vs Predicted")


with gr.Blocks(title="Linear Regression Demo") as demo:
    gr.Markdown("# Linear Regression: 2 Datasets")
    for key, cfg in DATASETS.items():
        with gr.Tab(cfg["title"]):
            build_tab(key)

if __name__ == "__main__":
    demo.launch()
