import re
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
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
            loader = YoutubeLoader(
                source,
                language=["en", "en-US"],
                add_video_info=False
            )
        elif source_type == "webpage":
            loader = WebBaseLoader(source)
        return loader.load()
    except Exception as exp:
        return f"Error occured: {exp}"
    
def create_chunks(document, chunk_size=1500):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(document)
    return chunks

def get_prompts():
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please identify the main themes and pay attention to the following when creating this summary:
    1. Capture the essence of the document by focusing on the main ideas and key details. 
    2. Avoid including unnecessary, meaningless and long explanations.
    3. Make the summary easy to read and understand by presenting it in a well-structured paragraph.
    Helpful Answer:"""

    map_prompt = PromptTemplate.from_template(map_template)

    reduce_template = """The following is set of summaries:
    {docs}
    Take the summaries given and directly analyse them to create a final summary. Pay attention to the following when creating this summary:

    1. Your beginning sentence will be "Here is the summary of the document:" 
    2. Start with a small general statement and then go item by item to declare main key points. 
    3. If there is advertising or promotion in the summaries, never mention the brands and offers here.
    4. Avoid using complex words and use simple sentences that are easy for the user to understand.
    5. Finish summarization with "Thank you." to indicate the completion of the task.
    Helpful Answer:"""

    reduce_prompt = PromptTemplate.from_template(reduce_template)

    return map_prompt, reduce_prompt

def get_chain(model):
    llm = Ollama(model=model)
    map_prompt, reduce_prompt = get_prompts()
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(
       llm_chain=reduce_chain, 
       document_variable_name="docs", 
    )

    reduce_documents_chain = ReduceDocumentsChain(
        # Final chain that is called
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    return map_reduce_chain

def get_summarization(source, source_type):
    transcript = load_document(source, source_type)
    chunks = create_chunks(transcript)
    chain = get_chain("llama3:instruct")
    summary = chain.invoke(chunks)

    return summary["output_text"]