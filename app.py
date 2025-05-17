from flask import Flask, render_template, request, jsonify
import os
import requests
import json
from bs4 import BeautifulSoup
# Updated imports for langchain
from langchain_community.embeddings import SentenceTransformerEmbeddings
# If you install langchain-huggingface, you can use this updated import:
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
# Import ChatOllama instead of LlamaCpp
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document
from langchain.prompts import PromptTemplate

app = Flask(__name__)

# --- Configuration ---
# Get API key from environment variable or use default for development
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "26fafe1d19be94686816682af49236ff5538ad76") # Replace with your actual key or ensure env var is set
# --- Ollama Configuration ---
# Specify the name of the Ollama model you want to use (e.g., mistral, llama2, phi3)
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "mistral") 
# Optional: Specify the base URL for the Ollama server if it's not localhost:11434
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Setup callback manager for server-side logging (optional with Ollama API)
# It won't stream to the client in this basic setup, but shows progress in server console.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def search_with_serper(query, gl="us", hl="en", num_results=7):
    """
    Perform a search using the Serper API
    
    Args:
        query (str): The search query
        gl (str): Geolocation code (e.g., 'us', 'in')
        hl (str): Host language (e.g., 'en')
        num_results (int): Number of results to return
    
    Returns:
        dict: The search results or None if error occurs
    """
    url = "https://google.serper.dev/search"
    
    payload = json.dumps({
        "q": query,
        "gl": gl,
        "hl": hl,
        "num": num_results
    })
    
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    if not SERPER_API_KEY or SERPER_API_KEY == "YOUR_SERPER_API_KEY":
         print("WARNING: SERPER_API_KEY is not set. Search will fail.")
         return None
         
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error making Serper API request: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding Serper API JSON response")
        return None

def get_urls_from_serper(query: str, num_results: int = 7) -> list[str]:
    """Uses Serper API to get search result URLs."""
    results = search_with_serper(query, gl="us", hl="en", num_results=num_results)
    if not results:
        print("Serper search failed or returned no results.")
        return []
        
    urls = []
    
    # Get URLs from organic results
    if 'organic' in results:
        urls.extend([res['link'] for res in results['organic'] if 'link' in res])
        
    # Get URL from answer box if available - often redundant but good practice
    if 'answerBox' in results and 'link' in results['answerBox']:
         # Check if the link is already in organic results to avoid immediate duplicates
         if results['answerBox']['link'] not in urls:
             urls.append(results['answerBox']['link'])

    # Get URL from related questions answer box if available
    if 'relatedQuestions' in results:
        for rq in results['relatedQuestions']:
            if 'answer' in rq and 'link' in rq['answer']:
                 if rq['answer']['link'] not in urls:
                     urls.append(rq['answer']['link'])

    # Get URL from knowledge graph if available
    if 'knowledgeGraph' in results and 'source' in results['knowledgeGraph'] and 'link' in results['knowledgeGraph']['source']:
         if results['knowledgeGraph']['source']['link'] not in urls:
             urls.append(results['knowledgeGraph']['source']['link'])

    # Remove duplicates while preserving order (more robust)
    urls = list(dict.fromkeys(urls))
    print(f"Found {len(urls)} URLs from search: {urls}")
    # Optionally filter out known bad domains or file types
    urls = [url for url in urls if not url.endswith(('.pdf', '.zip', '.jpg', '.png')) and 'youtube.com' not in url]
    print(f"Filtered URLs: {urls}")
    return urls


def fetch_and_process_url(url: str) -> list[Document]:
    """Fetches content from a single URL, extracts text, and chunks it."""
    try:
        print(f"Fetching content from {url}")
        # Add a reasonable timeout and user-agent
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Try lxml parser first, fall back to html.parser
        try:
            soup = BeautifulSoup(response.content, 'lxml')
        except Exception as e:
             print(f"lxml parser failed for {url}: {e}, falling back to html.parser")
             soup = BeautifulSoup(response.content, 'html.parser')

        # --- Improved Text Extraction ---
        # Remove script and style tags
        for script_or_style in soup(['script', 'style', 'meta', 'noscript', 'header', 'footer', 'nav', 'aside', 'form']):
            script_or_style.decompose()

        # Get text from the body and normalize whitespace
        text = soup.body.get_text(separator=' ', strip=True) if soup.body else ''
        text = ' '.join(text.split())

        # If text is still too short, might indicate an issue or very sparse page
        if not text.strip() or len(text) < 200:
            print(f"Not enough content after cleaning for {url}")
            return []

        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""] # Prioritize splitting by paragraphs, then lines, then sentences, etc.
        )
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        print(f"Created {len(chunks)} chunks from {url}")
        
        # Create Document objects
        docs = [Document(page_content=chunk, metadata={"source": url}) for chunk in chunks]
        return docs

    except requests.exceptions.Timeout:
        print(f"Timeout fetching {url}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error processing {url}: {e}")
        return []


def fetch_and_process_urls(urls: list[str]) -> list[Document]:
    """Fetches content from multiple URLs, extracts text, and chunks it."""
    all_docs = []
    
    # Process a limited number of URLs to avoid excessive fetching
    # For web search, 5-7 good sources are usually sufficient
    urls_to_process = urls[:7] # Limit to first 7 URLs
    
    for url in urls_to_process:
        docs = fetch_and_process_url(url)
        all_docs.extend(docs)
        
    print(f"Total documents created: {len(all_docs)}")
    return all_docs

def create_vector_store(docs: list[Document]):
    """Creates a Chroma vector store from documents using Sentence Transformers."""
    if not docs:
        print("No documents to create vector store from.")
        return None
        
    print(f"Creating vector store with {len(docs)} documents")
    try:
        # Initialize embeddings model
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        # If using langchain-huggingface:
        # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # Create vector store from documents
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
        print("Vector store created successfully.")
        return vectorstore
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def setup_rag_chain(vectorstore):
    """Sets up the RAG chain using the vector store and Ollama model."""
    if vectorstore is None:
        print("Cannot setup RAG chain without a valid vector store.")
        return None

    try:
        print(f"Loading Ollama model '{OLLAMA_MODEL_NAME}' from {OLLAMA_BASE_URL}")
        # Use ChatOllama for instruct models like Mistral
        llm = ChatOllama(
            model=OLLAMA_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            # Ollama parameters often map like:
            # max_tokens -> num_predict
            # n_ctx -> num_ctx
            num_predict=1000, # Max tokens to generate
            num_ctx=4096,     # Context window size (adjust based on your model's capability)
            # Optional: Enable streaming callbacks for server-side logging
            # callbacks=callback_manager, # Handled by RetrievalQA chain if streaming=True (not doing streaming here)
            # verbose=False # Suppress detailed logging from Langchain components if needed
        )
        print("Ollama model loaded successfully.")

    except Exception as e:
        print(f"Error loading Ollama model or connecting to server: {e}")
        print(f"Please ensure Ollama server is running at {OLLAMA_BASE_URL}")
        print(f"Also ensure the model '{OLLAMA_MODEL_NAME}' is pulled (run 'ollama pull {OLLAMA_MODEL_NAME}' in your terminal)")
        return None

    # Setup the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant chunks

    # Prompt template that instructs the model to cite sources
    # This template format is compatible with Mistral/Llama Instruct style
    template = """[INST] Use the following pieces of context to answer the user's question.
If you don't know the answer based on the context provided, just say that you don't know.
Do not make up an answer.
Cite the source URL from the context for each piece of information you use in your answer. Format citations like [URL].
Do not include the citation format [URL] if no relevant context is found or used for a specific part of the answer.
If the answer is a direct quote or heavily reliant on specific chunks, cite those chunks.

Context:
{context}

User question: {question} [/INST]"""

    custom_rag_prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Setup the RAG chain
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # 'stuff' fits all context into one prompt (suitable for smaller contexts)
            retriever=retriever,
            return_source_documents=True, # Return source documents for citation
            chain_type_kwargs={"prompt": custom_rag_prompt}
            # Set streaming=True here if you wanted token-by-token output (requires backend changes)
        )
        print("RAG chain setup complete.")
        return qa_chain
        
    except Exception as e:
        print(f"Error setting up RetrievalQA chain: {e}")
        return None


def answer_question_with_web_search(query: str):
    """Combines all steps to answer a question using web search RAG."""
    print(f"\n--- Processing query: {query} ---")
    
    # 1. Get URLs from Serper
    urls = get_urls_from_serper(query)
    if not urls:
        return {"answer": "Sorry, I couldn't find enough relevant information online.", "sources": []}

    # 2. Fetch and process content
    documents = fetch_and_process_urls(urls)
    if not documents:
        # We found URLs, but couldn't get usable content from them
        return {"answer": "Sorry, I found some websites but couldn't extract useful information from them.", "sources": urls}

    # 3. Create Vector Store
    vectorstore = create_vector_store(documents)
    if vectorstore is None:
         return {"answer": "Sorry, I couldn't process the content to set up the knowledge base.", "sources": urls}

    # 4. Setup RAG Chain
    qa_chain = setup_rag_chain(vectorstore)
    if qa_chain is None:
        # setup_rag_chain already prints specific error about Ollama
        return {"answer": "Failed to initialize the AI model. Please check the server logs.", "sources": urls}

    # 5. Run the RAG chain
    try:
        print("Running RAG chain to generate answer")
        # Use a dictionary input for the query
        result = qa_chain({"query": query})
        
        answer = result.get("result", "No answer generated by the model.")
        source_documents = result.get("source_documents", [])

        # Extract and deduplicate sources from source_documents metadata
        sources = sorted(list(set([doc.metadata['source'] for doc in source_documents if 'source' in doc.metadata])))
        
        print(f"Answer generated. Used {len(source_documents)} source chunks from {len(sources)} unique sources.")
        print(f"Sources used: {sources}")

        return {"answer": answer, "sources": sources}
        
    except Exception as e:
        print(f"Error in RAG chain execution: {e}")
        # Provide a more specific error message about the LLM if possible
        return {"answer": f"An error occurred while generating the answer. Please check the Ollama server and model '{OLLAMA_MODEL_NAME}'. Error: {str(e)}", "sources": urls}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    result = answer_question_with_web_search(question)
    return jsonify(result)

# Custom error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    # Log the actual error in the server logs for debugging
    app.logger.exception('An internal server error occurred')
    return jsonify({"error": "Internal server error. Please check server logs."}), 500

if __name__ == "__main__":
    print("--- Starting RAG Flask Application ---")
    print(f"Using Ollama Model: {OLLAMA_MODEL_NAME} at {OLLAMA_BASE_URL}")
    print(f"Using Embedding Model: {EMBEDDING_MODEL_NAME}")
    
    if not SERPER_API_KEY or SERPER_API_KEY == "YOUR_SERPER_API_KEY":
         print("\n!!! WARNING: SERPER_API_KEY is not set or is using the placeholder value.")
         print("!!! Web search functionality will not work.")
         print("!!! Set the SERPER_API_KEY environment variable or replace the placeholder.")
         
    print("\nAttempting to connect to Ollama server and load model...")
    # Basic check: Try initializing the ChatOllama instance.
    # This might not catch *all* errors (e.g., model not pulled) but is a start.
    # The actual error is more likely during the first call to the model.
    try:
        test_llm = ChatOllama(
            model=OLLAMA_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0 # Use temp=0 for a test call
        )
        # Optional: Make a tiny test call - more reliable but adds delay
        # test_llm.invoke("Hi", max_tokens=5)
        print("Ollama initialization successful. Model *might* be ready.")
        print("The first user query will perform a more thorough check.")
    except Exception as e:
        print(f"\n!!! ERROR: Failed to initialize Ollama model '{OLLAMA_MODEL_NAME}'.")
        print(f"!!! Please ensure the Ollama server is running at {OLLAMA_BASE_URL}.")
        print(f"!!! And that the model '{OLLAMA_MODEL_NAME}' is downloaded (run 'ollama pull {OLLAMA_MODEL_NAME}').")
        print(f"!!! Specific error: {e}")
        # Decide whether to exit or allow the app to start but fail on queries.
        # Allowing it to start might be useful for debugging other parts.
        # raise e # Uncomment to prevent server start if Ollama fails
    print("---------------------------------------")


    # Use environment variables for host/port if available
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 5000))
    # Debug mode should be False in production
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true" 
    
    print(f"Starting Flask app on http://{host}:{port}/ (debug: {debug})")
    app.run(host=host, port=port, debug=debug, use_reloader=False) # use_reloader=False often needed with LlamaCpp/Ollama