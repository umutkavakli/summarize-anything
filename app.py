import gradio as gr
from summarizer import get_summarization

models = ["Llama 2"]
sources = {
    "Youtube Video URL": "youtube", 
     "PDF": "pdf", 
    "Webpage URL": "webpage"
}


css = """
h1 {
    text-align: center;
    display: block;
}
"""

def update_source(source):
    youtube_visibility = (source == "Youtube Video URL")
    pdf_visibility = (source == "PDF")
    web_visibility = (source == "Webpage URL")
    return gr.Textbox(visible=youtube_visibility), gr.UploadButton(visible=pdf_visibility), gr.Textbox(visible=web_visibility)

def result(youtube, pdf, web, source_type):
    if source_type == "Youtube Video URL":
        return get_summarization(youtube, sources[source_type])
    elif source_type == "PDF":
        return get_summarization(pdf, sources[source_type])
    else:
        return get_summarization(web, sources[source_type])

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
            task = gr.Dropdown(choices=list(sources.keys()), label="Source", value=list(sources.keys())[0], interactive=True)
        with gr.Column():
            youtube_box = gr.Textbox(label="Youtube Video URL", visible=True)
            pdf_box = gr.UploadButton("Click to Upload a PDF File", file_types=["file"], file_count="single", visible=False)
            webpage_box = gr.Textbox(label="Webpage URL", visible=False)
            task.change(update_source, inputs=[task], outputs=[youtube_box, pdf_box, webpage_box])

            output = gr.Textbox(label="Summarization", lines=10)
            summarize_button = gr.Button("Summarize")
            summarize_button.click(fn=result, inputs=[youtube_box, pdf_box, webpage_box, task], outputs=[output])

# Launch the Gradio interface
demo.launch()
