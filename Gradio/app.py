import gradio as gr
from data import df

with gr.Blocks() as demo:
    gr.LinePlot(df, x="weight", y="height")

demo.launch()