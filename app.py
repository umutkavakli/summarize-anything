import gradio as gr
import youtube_summarizer
import pdf_summarizer 

models = ["Llama 2"]
sources = ["Youtube Video", "PDF", "Website"]

css = """
h1 {
    text-align: center;
    display: block;
}
"""

def update_source(source):
    youtube_visibility = (source == "Youtube Video")
    pdf_visibility = (source == "PDF")
    return gr.Textbox(visible=youtube_visibility), gr.UploadButton(visible=pdf_visibility)

def out(youtube, pdf, source):
    if source == "Youtube Video":
        return youtube_summarizer.get_summarization(youtube)
    else:
        return pdf_summarizer.get_summarization(pdf)

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
        # Summarize Anything
        If you are tired to watch a video or read blog/pdf to have a sense in main points of source, 
        this app will help you! You can summarize youtube videos, webpage contents and PDFs with only one click!
        """
    )

    with gr.Row():
        with gr.Column():
            model = gr.Dropdown(choices=models, label="Model", value=models[0], interactive=True)
            task = gr.Dropdown(choices=sources, label="Source", value=sources[0], interactive=True)
        with gr.Column():
            youtube = gr.Textbox(label="Youtube URL", visible=True)
            pdf = gr.UploadButton("Click to Upload a PDF File", file_types=["file"], file_count="single", visible=False)
            task.change(update_source, inputs=[task], outputs=[youtube, pdf])
            output = gr.Textbox(label="Summarization", lines=10)
            summarize_button = gr.Button("Summarize")
            summarize_button.click(fn=out, inputs=[youtube, pdf, task], outputs=[output])

# Launch the Gradio interface
demo.launch()
