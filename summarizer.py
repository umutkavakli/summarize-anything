import re
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import YoutubeLoader, PyPDFLoader, WebBaseLoader


def validate_source(source, source_type):
    if source_type == "pdf":
        return source.endswith(".pdf")
    elif source_type == "youtube":
        youtube_regex = r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$'
        return re.match(youtube_regex, source, re.IGNORECASE) is not None
    elif source_type == "webpage":
        webpage_regex = r'^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})(\/[^\s]*)?$'
        return re.match(webpage_regex, source, re.IGNORECASE) is not None
    else:
        pass

def load_document(source, source_type):
    if not validate_source(source, source_type):
        return "Invalid source. Please check and correct it."
    try:
        if source_type == "pdf":
            loader = PyPDFLoader(source)
        elif source_type == "youtube":
            loader = YoutubeLoader.from_youtube_url(
                source,
                language=["en", "en-US"],
                add_video_info=False
            )
        elif source_type == "webpage":
            loader = WebBaseLoader(source)
        document = loader.load()
        return document
    except Exception as exp:
        return f"Error occured: {exp}"
    
def create_chunks(document, chunk_size=1500):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(document)
    return chunks

def get_prompt():

    template = """The following is a document:
    {context}
    Take this document given and directly analyse them to create a final summary. Pay attention to the following when creating this summary:

    1. Your beginning sentence will be "Here is the summary of the document:" 
    2. Start with a small general statement and then go item by item to declare main key points. 
    3. If there is advertising or promotion in the summaries, never mention the brands and offers here.
    4. Avoid using complex words and use simple sentences that are easy for the user to understand.
    5. Finish summarization with "Thank you." to indicate the completion of the task.
    Helpful Answer:"""

    prompt = PromptTemplate.from_template(template)

    return prompt

def get_chain(model):
    llm = Ollama(model=model)
    prompt = get_prompt()
    chain = create_stuff_documents_chain(llm, prompt)

    return chain

def get_summarization(source, source_type):
    document = load_document(source, source_type)
    chunks = create_chunks(document)
    chain = get_chain("llama3:instruct")
    summary = chain.invoke({'context': chunks})

    return summary