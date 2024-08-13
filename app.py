import gradio as gr
import youtube_summarizer
import pdf_summarizer 
import webpage_summarizer

models = ["Llama 2"]
sources = ["Youtube Video", "PDF", "Webpage URL"]

css = """
h1 {
    text-align: center;
    display: block;
}
"""

def update_source(source):
    youtube_visibility = (source == "Youtube Video")
    pdf_visibility = (source == "PDF")
    web_visibility = (source == "Webpage URL")
    return gr.Textbox(visible=youtube_visibility), gr.UploadButton(visible=pdf_visibility), gr.Textbox(visible=web_visibility)

def out(youtube, pdf, web, source):
    if source == "Youtube Video":
        return youtube_summarizer.get_summarization(youtube)
    elif source == "PDF":
        return pdf_summarizer.get_summarization(pdf)
    else:
        return webpage_summarizer.get_summarization(web)

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
            web = gr.Textbox(label="Webpage URL", visible=False)
            task.change(update_source, inputs=[task], outputs=[youtube, pdf, web])
            output = gr.Textbox(label="Summarization", lines=10)
            summarize_button = gr.Button("Summarize")
            summarize_button.click(fn=out, inputs=[youtube, pdf, web, task], outputs=[output])

# Launch the Gradio interface
demo.launch()
