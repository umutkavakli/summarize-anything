import gradio as gr

from youtube_summarizer import get_summarization

models = ["Llama 2"]
sources = ["Youtube Video", "PDF", "Website"]

css = """
h1 {
    text-align: center;
    display: block;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
        # Summarize Anything
        If you are tired to watch a video or read blog/pdf to have a sense in main points of source, 
        this app will help you! You can summarize youtube videos, webpage contents and PDFs with only one click!
        """)


    with gr.Row():
        with gr.Column():
            model = gr.Dropdown(choices=models, label="Model", value=models[0], interactive=True)
            task = gr.Dropdown(choices=sources, label="Source", value=sources[0], interactive=True)

        with gr.Column():
            url = gr.Textbox(label="Youtube URL")
            output = gr.Textbox(label="Summarization", lines=10)
            summarize_btn = gr.Button("Summarize")

    @summarize_btn.click(inputs=url, outputs=output)
    def out(url):
        return get_summarization(url)
    
demo.launch()

