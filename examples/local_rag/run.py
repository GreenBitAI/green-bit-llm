import argparse
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from green_bit_llm.langchain import GreenBitPipeline, ChatGreenBit, GreenBitEmbeddings
import torch
import re

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Helper function to print task separators
def print_task_separator(task_name):
    print("\n" + "="*50)
    print(f"Task: {task_name}")
    print("="*50 + "\n")

def clean_output(text):
    # Remove all non-alphanumeric characters except periods and spaces
    cleaned = re.sub(r'[^a-zA-Z0-9\.\s]', ' ', text)
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Remove any mentions of "assistant" or other unwanted words
    cleaned = re.sub(r'\b(assistant|correct|I apologize|mistake)\b', '', cleaned, flags=re.IGNORECASE)
    # Remove any remaining leading/trailing whitespace
    cleaned = cleaned.strip()
    # Ensure the first letter is capitalized
    cleaned = cleaned.capitalize()
    # Ensure the answer ends with a period
    if cleaned and not cleaned.endswith('.'):
        cleaned += '.'
    return cleaned

def extract_answer(text):
    # Try to extract a single sentence answer
    match = re.search(r'([A-Z][^\.!?]*[\.!?])', text)
    if match:
        return match.group(1)
    # If no clear sentence is found, return the first 100 characters
    return text[:100] + '...' if len(text) > 100 else text

# Load and prepare data
def prepare_data(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    return all_splits

# Create vector store
def create_vectorstore(documents, embedding_model):
    model_kwargs = {'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}

    greenbit_embeddings = GreenBitEmbeddings.from_model_id(
        model_name=embedding_model,
        cache_dir="cache",
        multi_process=False,
        show_progress=False,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return Chroma.from_documents(documents=documents, embedding=greenbit_embeddings)

# Initialize GreenBit model
def init_greenbit_model(model_id, max_tokens):
    pipeline = GreenBitPipeline.from_model_id(
        model_id=model_id,
        model_kwargs={"dtype": torch.half, "seqlen": 2048, "device_map": "auto"},
        pipeline_kwargs={"max_new_tokens": max_tokens, "temperature": 0.7, "do_sample": True},
    )

    return ChatGreenBit(llm=pipeline)


# Task 1: Rap Battle Simulation
def simulate_rap_battle(model):
    print_task_separator("Rap Battle Simulation")
    prompt = "Simulate a rap battle between rag and graphRag."
    response = model.invoke(prompt)
    print(response.content)


# Task 2: Summarization
def summarize_docs(model, vectorstore, question):
    print_task_separator("Summarization")
    prompt_template = "Summarize the main themes in these retrieved docs in a single, complete sentence of no more than 200 words: {docs}"
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
            {"docs": format_docs}
            | prompt
            | model
            | StrOutputParser()
            | clean_output
            | extract_answer
    )
    docs = vectorstore.similarity_search(question)
    response = chain.invoke(docs)
    print(response)


# Task 3: Q&A
def question_answering(model, vectorstore, question):
    print_task_separator("Q&A")
    RAG_TEMPLATE = """
    Answer the following question based on the context provided. Give a direct and concise answer in a single, complete sentence of no more than 100 words. Do not include any additional dialogue or explanation.

    Context:
    {context}

    Question: {question}

    Answer:"""

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    chain = (
            RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
            | rag_prompt
            | model
            | StrOutputParser()
            | clean_output
            | extract_answer
    )
    docs = vectorstore.similarity_search(question)
    response = chain.invoke({"context": docs, "question": question})
    print(response)


# Task 4: Q&A with Retrieval
def qa_with_retrieval(model, vectorstore, question):
    print_task_separator("Q&A with Retrieval")
    RAG_TEMPLATE = """
    Answer the following question based on the retrieved information. Provide a direct and concise answer in a single, complete sentence of no more than 100 words. Do not include any additional dialogue or explanation.

    Question: {question}

    Answer:"""

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    retriever = vectorstore.as_retriever()
    qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | model
            | StrOutputParser()
            | clean_output
            | extract_answer
    )
    response = qa_chain.invoke(question)
    print(response)



def main(model_id, embedding_model, query, max_tokens, web_source):
    print_task_separator("Initialization")
    print("Preparing data and initializing model...")
    # Prepare data and initialize model
    all_splits = prepare_data(web_source)
    vectorstore = create_vectorstore(all_splits, embedding_model)
    model = init_greenbit_model(model_id, max_tokens)
    print("Initialization complete.")

    # Execute tasks
    simulate_rap_battle(model)
    summarize_docs(model, vectorstore, query)
    question_answering(model, vectorstore, query)
    qa_with_retrieval(model, vectorstore, query)

    print("\nAll tasks completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLP tasks with specified model, query, max tokens, and web source.")
    parser.add_argument("--model", type=str, default="GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0", help="Model ID to use for the tasks")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L12-v2", help="Embedding model to use for vector store creation")
    parser.add_argument("--query", type=str, required=True, help="Query to use for the tasks")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens for model output (default: 200)")
    parser.add_argument("--web_source", type=str, default="https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/",
                        help="URL of the web source to load data from (default: Microsoft Research blog post on GraphRAG)")
    args = parser.parse_args()

    main(args.model, args.embedding_model, args.query, args.max_tokens, args.web_source)