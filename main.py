import streamlit as st
import os
import re
import base64
import json
import tempfile
import html
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from utils.helpers import get_base64_of_background_image, get_context, format_message_content, process_latex
import yaml

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#------------------------------------------------------------------------------
# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
background_image_path = "images/Theme.jpg"
background_image = get_base64_of_background_image(background_image_path)
# Loading the Config file
with open("config/hyperparams.yaml", "r") as f:
    config=yaml.safe_load(f)
PDFS_DIRECTORY = config['pdfs_directory']
DB_PATH = config['db_path']
EMBEDDING_MODEL = config['embedding_model']
CHUNK_SIZE = config['text_splitting']['chunk_size']
CHUNK_OVERLAP = config['text_splitting']['chunk_overlap']
ADD_START_INDEX = config['text_splitting']['add_start_index']
SEARCH_TYPE = config['retrieval']['search_type']
TOP_K_RESULTS = config['retrieval']['top_k']
MIN_SIMILARITY = config['retrieval'].get('min_similarity', 0.0)
MAX_FILE_SIZE_MB = config['file_upload_limits']['max_file_size_mb']
MAX_FILE_SIZE_BYTES = config['file_upload_limits']['max_file_size_bytes']
MAX_HISTORY_MESSAGES = config['chat_history']['max_history_messages']  # Number of messages to include in context (5 exchanges)

#------------------------------------------------------------------------------
# Model Configuration
AVAILABLE_MODELS = {
    "Qwen3 32B": "qwen/qwen3-32b",
    "Kimi K2 Instruct": "moonshotai/kimi-k2-instruct",
    "GPT-OSS 120B": "openai/gpt-oss-120b",
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Gemma3 1B (Ollama)": "ollama:gemma3:1b",
    "Qwen3 4B": "ollama:qwen3:4b",
}

# Routing Prompt - decides if retrieval is needed
ROUTING_PROMPT = '''
You are a routing assistant. Your job is to determine if a question requires retrieving information from documents or if it can be answered directly.

Analyze the question and respond with ONLY "RETRIEVE" or "DIRECT" based on these rules:

RETRIEVE if:
- The question asks about specific content, data, or facts from documents
- The question requires detailed information that would be in research papers or PDFs
- The question asks to explain, summarize, or analyze document content
- The question mentions specific topics that require document context
- The question asks about mathematical derivations, formulas, or technical details from papers
- The question is asking "what does the paper say about..."

DIRECT if:
- The question is a general greeting (hi, hello, how are you)
- The question asks about your capabilities or who you are
- The question is general knowledge that doesn't require specific document content
- The question is a follow-up clarification that doesn't need new document retrieval
- The question is asking for general explanations of well-known concepts

Question: {question}

Response (RETRIEVE or DIRECT):'''

# Main Prompt Template for answering with documents
PROMPT_TEMPLATE_WITH_DOCS = '''
You are Helpful assistant named ResearchGPT. You are given the following extracted parts of a long document, chat history, and a question. Provide a conversational answer based on the context provided and previous conversation.
Basically, you are An expert in scientific research papers. Use the context to answer the question as accurately as possible.
Your knowledge is like an university professor with expertise in research papers. Who explains everything clearly. And if you are asked to do any math you always provide the mathematical equations in latex format.
Your solution Generation Format: 
Step1: Give all the necessary definitions needed. 
Step2: Explain the solution step by step in detail. If mathematical equations are there then try to derive them step by step. As you are a high level professor.
Always format mathematical equations in LaTeX format.
Step3: Finally provide the final TLDR; summarized form of the answer.

Chat History: {chat_history}
Context: {context}
Question: {question}
Answer: '''

# Prompt Template for answering without documents
PROMPT_TEMPLATE_WITHOUT_DOCS = '''
You are ResearchGPT, a helpful AI assistant specialized in research and scientific topics.
You can engage in general conversation and answer questions based on your knowledge.
When answering, be conversational and helpful.

Chat History: {chat_history}
Question: {question}
Answer: '''

#------------------------------------------------------------------------------

# Page Configuration
st.set_page_config(page_title="Research GPT",page_icon="🔬",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# MathJax Support for LaTeX Rendering
st.markdown("""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
                inlineMath: [['$','$'], ['\\(','\\)']],
                displayMath: [['$$','$$'], ['\\[','\\]']],
                processEscapes: true
            },
            CommonHTML: { linebreaks: { automatic: true } },
            "HTML-CSS": { scale: 100 }
        });
    </script>
""", unsafe_allow_html=True)

#------------------------------------------------------------------------------
# Model Initialization
llm_instances = {}
for model_name, model_id in AVAILABLE_MODELS.items():
    if model_id.startswith("ollama:"):
        # Extract the actual model name after "ollama:"
        ollama_model = model_id.split("ollama:")[1]
        llm_instances[model_name] = ChatOllama(model=ollama_model)
    else:
        llm_instances[model_name] = ChatGroq(model=model_id, api_key=groq_api_key)

#------------------------------------------------------------------------------
class VectorDB:
    def __init__(self, pdfs_directory=PDFS_DIRECTORY, db_path=DB_PATH):
        self.pdfs_directory = pdfs_directory
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.faiss_db = None
        self.vectorized_files = set()  # Track which files are vectorized
        
    def get_pdf_files(self):
        """Get list of PDF files in the directory"""
        if not os.path.exists(self.pdfs_directory):
            os.makedirs(self.pdfs_directory)
            return []
        
        return [f for f in os.listdir(self.pdfs_directory) if f.endswith(".pdf")]

    def process_pdfs(self, additional_pdf_paths=None, force_rebuild=False):
        """Process PDFs including optional uploaded PDFs"""
        try:
            # Check if we need to rebuild
            current_files = set(self.get_pdf_files())
            
            # Load existing database if available and not forcing rebuild
            if not force_rebuild and os.path.exists(os.path.join(self.db_path, 'index.faiss')):
                try:
                    self.faiss_db = FAISS.load_local(
                        self.db_path, 
                        self.embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    
                    # Check if we have new files to add
                    if additional_pdf_paths or current_files != self.vectorized_files:
                        documents = []
                        
                        # Add new PDFs from directory
                        new_files = current_files - self.vectorized_files
                        for filename in new_files:
                            file_path = os.path.join(self.pdfs_directory, filename)
                            loader = PDFPlumberLoader(file_path)
                            docs = loader.load()
                            documents.extend(docs)
                        
                        # Add uploaded PDFs
                        if additional_pdf_paths:
                            for pdf_path in additional_pdf_paths:
                                if os.path.exists(pdf_path):
                                    loader = PDFPlumberLoader(pdf_path)
                                    docs = loader.load()
                                    documents.extend(docs)
                        
                        if documents:
                            # Create chunks and add to existing database
                            text_chunks = self.create_chunks(documents)
                            self.faiss_db.add_documents(text_chunks)
                            self.faiss_db.save_local(self.db_path)
                            self.vectorized_files.update(current_files)
                    
                    return self.faiss_db
                except Exception as e:
                    st.warning(f"Could not load existing database: {str(e)}. Rebuilding...")
                    force_rebuild = True
            
            # Build from scratch
            documents = self.load_pdfs()
            
            # Add uploaded PDFs if provided
            if additional_pdf_paths:
                for pdf_path in additional_pdf_paths:
                    if os.path.exists(pdf_path):
                        loader = PDFPlumberLoader(pdf_path)
                        uploaded_docs = loader.load()
                        documents.extend(uploaded_docs)
            
            if not documents:
                st.warning("No documents found to vectorize!")
                return None
                
            text_chunks = self.create_chunks(documents)
            self.faiss_db = FAISS.from_documents(text_chunks, self.embeddings)
            os.makedirs(self.db_path, exist_ok=True)
            self.faiss_db.save_local(self.db_path)
            self.vectorized_files = current_files
            
            return self.faiss_db
            
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            return None

    def load_pdfs(self):
        """Load PDFs from the pdfs directory"""
        documents = []
        if not os.path.exists(self.pdfs_directory):
            os.makedirs(self.pdfs_directory)
            return documents

        for filename in os.listdir(self.pdfs_directory):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.pdfs_directory, filename)
                try:
                    loader = PDFPlumberLoader(file_path)
                    doc = loader.load()
                    documents.extend(doc)
                except Exception as e:
                    st.warning(f"Could not load {filename}: {str(e)}")
        return documents

    def create_chunks(self, documents):
        text_splitter = TokenTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=ADD_START_INDEX,
            encoding_name="cl100k_base"
        )
        return text_splitter.split_documents(documents)

    def get_retriever(self):
        """Get retriever, loading database if not already loaded"""
        if not self.faiss_db:
            if os.path.exists(os.path.join(self.db_path, 'index.faiss')):
                try:
                    self.faiss_db = FAISS.load_local(
                        self.db_path, 
                        self.embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    # Update vectorized files tracking
                    self.vectorized_files = set(self.get_pdf_files())
                except Exception as e:
                    st.error(f"Error loading vector database: {str(e)}")
                    return None
            else:
                return None
        
        return self.faiss_db.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k": TOP_K_RESULTS})
    
    def is_db_loaded(self):
        """Check if database is loaded and ready"""
        return self.faiss_db is not None

vector_db = VectorDB()

# Initialize vector database on startup if PDFs exist
if os.path.exists(os.path.join(DB_PATH, 'index.faiss')):
    try:
        vector_db.faiss_db = FAISS.load_local(
            DB_PATH, 
            vector_db.embeddings, 
            allow_dangerous_deserialization=True
        )
        vector_db.vectorized_files = set(vector_db.get_pdf_files())
    except Exception as e:
        st.warning(f"Could not load existing database on startup: {str(e)}")

#------------------------------------------------------------------------------
# Helpers for Chat Interface

def decide_retrieval(query, model):
    """
    Use the model to decide if retrieval is needed for the query
    Returns: 'RETRIEVE' or 'DIRECT'
    """
    try:
        routing_prompt = ChatPromptTemplate.from_template(ROUTING_PROMPT)
        chain = routing_prompt | model
        
        result = chain.invoke({"question": query})
        
        # Extract the decision
        if hasattr(result, 'content'):
            decision = result.content.strip().upper()
        else:
            decision = str(result).strip().upper()
        
        # Ensure we only return valid decisions
        if 'RETRIEVE' in decision:
            return 'RETRIEVE'
        elif 'DIRECT' in decision:
            return 'DIRECT'
        else:
            # Default to RETRIEVE if unclear
            return 'RETRIEVE'
            
    except Exception as e:
        st.warning(f"Error in routing decision: {str(e)}. Defaulting to RETRIEVE.")
        return 'RETRIEVE'


def retrieve_docs(query):
    """Retrieve relevant documents for the query"""
    retriever = vector_db.get_retriever()
    
    if retriever is None:
        return []

    try:
        # Use similarity_search_with_score for FAISS
        if hasattr(vector_db.faiss_db, "similarity_search_with_score"):
            docs_with_scores = vector_db.faiss_db.similarity_search_with_score(query, k=TOP_K_RESULTS)
            
            # Filter based on threshold (note: FAISS uses distance, lower is better)
            # For L2 distance, we might want to invert the logic or adjust threshold
            filtered_docs = [doc for doc, score in docs_with_scores if score >= MIN_SIMILARITY]
            
            # If no document meets threshold, return all retrieved docs
            if not filtered_docs and docs_with_scores:
                filtered_docs = [doc for doc, score in docs_with_scores]
            
            return filtered_docs

        # Fallback for non-FAISS retrievers
        if hasattr(retriever, 'get_relevant_documents'):
            return retriever.get_relevant_documents(query)

        if hasattr(retriever, 'invoke'):
            return retriever.invoke(query)
            
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

    return []


def format_chat_history(chat_history):
    if not chat_history:
        return "No previous conversation."

    formatted = []
    for message in chat_history[-MAX_HISTORY_MESSAGES:]:  # Use global parameter
        role = "User" if message['role'] == 'user' else "Assistant"
        formatted.append(f"{role}: {message['content']}")
    return "\n".join(formatted)


def answer_query_with_docs(documents, model, query, chat_history):
    """Answer query using retrieved documents"""
    context = get_context(documents)
    history_text = format_chat_history(chat_history)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_WITH_DOCS)
    chain = prompt | model
    
    out = chain.invoke({
        "question": query,
        "context": context,
        "chat_history": history_text
    })
    
    if hasattr(out, 'content'):
        return out.content
    return out


def answer_query_direct(model, query, chat_history):
    """Answer query directly without document retrieval"""
    history_text = format_chat_history(chat_history)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_WITHOUT_DOCS)
    chain = prompt | model
    
    out = chain.invoke({
        "question": query,
        "chat_history": history_text
    })
    
    if hasattr(out, 'content'):
        return out.content
    return out

#------------------------------------------------------------------------------
# STYLING
# Custom CSS with combined styles and math support (unchanged visuals)
GLASS_CONTAINER_BG = "rgba(255, 255, 255, 0.1)"
st.markdown(f"""
    <style>

    .stApp {{
        background-image: url("data:image/jpg;base64,{background_image}");
        background-size: cover;
    }}
    .main .block-container {{
        max-width: 2000px;   /* set your custom width */
        padding-left: 2rem;
        padding-right: 2rem;
    }}
    
    .glass-container {{
        background: {GLASS_CONTAINER_BG};
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 20px;
        margin: 10px 0;
    }}
    .chat-message {{
        padding: 1.5rem;
        margin: 8px 0;
        border-radius: 12px;
        display: flex;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }}

    /* Assistant bubble - full width with word wrapping */
    .assistant-message {{
        background: rgba(0, 0, 0, 0.5);
        width: 100%;
        justify-content: flex-start;
        margin-bottom: 100px;
        overflow-wrap: break-word;
        word-break: break-word;
    }}

    /* User bubble - small and right-aligned */
    .user-message {{
        background: rgba(0, 0, 0, 0.5);
        max-width: 40%;        /* make user bubble compact */
        margin-left: auto;     /* pushes it to the right */
        text-align: right;
        float: right;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }}
    
    .message-content {{
        color: #FFFFFF;
        width: 100%;
        max-width: 100%;
        overflow-wrap: break-word;
        word-break: break-word;
        white-space: pre-wrap;
    }}
    
    .math-container {{
        background-color: rgba(64, 65, 79, 0.7);
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        overflow-x: auto;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    .inline-math {{
        background-color: rgba(64, 65, 79, 0.5);
        border-radius: 4px;
        padding: 0.2rem 0.4rem;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .bullet-glass-container {{
        background-color: rgba(64, 65, 79, 0.7);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    .sidebar .block-container {{
        background: rgba(45, 45, 45, 0.7);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 10px;
    }}
    section[data-testid="stSidebar"] {{
        background: rgba(0, 0, 0, 0.3);
        width: 300px;
    }}
    .uploadedFile {{
        background-color: rgba(68, 70, 84, 0.7);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid rgba(86, 88, 105, 0.7);
    }}
    /* Enhanced markdown styling */
    .chat-message p {{
        margin: 0;
        padding: 0;
        word-wrap: break-word;
    }}
    .chat-message strong {{
        font-weight: bold;
        color: rgb(255, 153, 0);
    }}
    .chat-message em {{
        font-style: italic;
        color: #ADD8E6;
    }}
    .chat-message code {{
        background: rgba(0, 0, 0, 0.3);
        padding: 2px 4px;
        border-radius: 4px;
        font-family: monospace;
        word-break: break-all;
    }}
    .has-jax {{
        font-size: 100%;
    }}
    /* Compact button styling */
    .stButton > button {{
        padding: 0.35rem 0.75rem;
        font-size: 0.85rem;
        height: 2rem;
        line-height: 1.2;
    }}
    div[data-testid="stFileUploader"] {{
        padding: 0.5rem 0;
    }}
    div[data-testid="stFileUploader"] > div {{
        padding: 0.5rem;
    }}
    
    /* Chat input styling - Large square box with black interior and orange glow */
    .stChatInput {{
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    
    .stChatInput > div {{
        background: transparent !important;
        padding: 0 !important;
    }}
    
    div[data-testid="stChatInput"] {{
        background: transparent !important;
        padding: 2rem 0 !important;
        margin: 0 !important;
        transform: translateY(0px) !important;  
        position: fixed !important;
        z-index: -999 !important;
        bottom: 0 !important;
    }}
    
    div[data-testid="stChatInput"] > div {{
        background: transparent !important;
        padding: 0 !important;
    }}
    
    div[data-testid="stChatInput"] > div > div {{
        background: transparent !important;
    }}
    
    div[data-testid="stChatInput"] form {{
        background: #000000 !important;
        border: 2px solid rgba(255, 140, 60, 0.6) !important;
        border-radius: 0px !important;
        padding: 0 !important;
        box-shadow: 0 0 20px rgba(255, 140, 60, 0.5), 
                    0 0 40px rgba(255, 140, 60, 0.3),
                    0 0 60px rgba(255, 140, 60, 0.2) !important;
        min-height: 180px !important;

    }}
    
    div[data-testid="stChatInput"] form > div {{
        background: #000000 !important;
        padding: 2rem 2rem !important;
        min-height: 100px !important;
        display: flex !important;
        align-items: center !important;
        
    }}
    
    div[data-testid="stChatInput"] textarea {{
        background: rgba(80, 80, 80, 0.4) !important;
        color: #FFFFFF !important;
        font-size: 1.2rem !important;
        min-height: 140px !important;
        max-height: 400px !important;
        padding: 1rem !important;
        border: none !important;
        outline: none !important;
        resize: none !important;
        line-height: 1.6 !important;
    }}
    
    div[data-testid="stChatInput"] textarea::placeholder {{
        color: rgba(255, 255, 255, 0.4) !important;
    }}
    
    div[data-testid="stChatInput"] button {{
        background: transparent !important;
        color: rgba(255, 140, 60, 0.6) !important;
        border: none !important;
        padding: 1rem !important;
        margin: 0 !important;
        font-size: 1.2rem !important;
    }}
    
    div[data-testid="stChatInput"] button:hover {{
        color: rgba(255, 140, 60, 1) !important;
        background: rgba(255, 140, 60, 0.1) !important;
    }}
    
    /* Remove any outer container backgrounds */
    .stChatInput, 
    .stChatInput * {{
        box-sizing: border-box !important;
    }}
    
    div[data-testid="stBottom"] {{
        background: transparent !important;
    }}
    
    div[data-testid="stBottom"] > div {{
        background: transparent !important;
    }}
    </style>
""", unsafe_allow_html=True)

#------------------------------------------------------------------------------
# SESSION STATE INITIALIZATION

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_db_status' not in st.session_state:
    st.session_state.vector_db_status = os.path.exists(os.path.join(DB_PATH, 'index.faiss'))
if 'uploaded_pdf_paths' not in st.session_state:
    st.session_state.uploaded_pdf_paths = []

#------------------------------------------------------------------------------
# MAIN APP INTERFACE
# Display header
st.markdown(
    """
    # 🔬 Research GPT  
    **Upload any PDFs to get started!**  
    """
)

# st.markdown(
#     """
#     <div style="margin-top: -35px; margin-bottom: 20px; margin-left: 10px;">
#         <h1 style="margin: 0; padding: 0;">🔬 Research GPT</h1>
#         <p style="margin: 0; padding: 0;"><strong>Upload any PDFs to get started!</strong></p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )
# ------------------------------------------------------------------------------
# SIDEBAR
with st.sidebar:
    st.image("images/Theam.jpg", use_container_width=True)

    # Model Selection
    st.markdown("<h3 style='color: #ECECF1;'>🤖 Select Model</h3>", unsafe_allow_html=True)
    llm_choice = st.selectbox("Choose Language Model", list(AVAILABLE_MODELS.keys()), help="Select the AI model for analysis")

    # Get selected LLM instance
    llm = llm_instances[llm_choice]

    st.markdown("---")

    # PDF Upload Section
    st.markdown("<h3 style='color: #ECECF1;'> Upload PDFs</h3>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        f"Upload PDFs (Max {MAX_FILE_SIZE_MB}MB each)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload multiple PDFs to add to the vector database temporarily. Will be removed on page refresh."
    )

    if uploaded_files:
        st.session_state.uploaded_pdf_paths = []
        valid_files = []

        for uploaded_file in uploaded_files:
            # Check file size using global parameter
            if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                st.error(f"ERROR: {uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit!")
            else:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    st.session_state.uploaded_pdf_paths.append(tmp_file.name)
                    valid_files.append(uploaded_file.name)

    # Vectorize PDFs Button
    if st.button("🔄 Vectorize PDFs", use_container_width=True):
        with st.status("Processing PDFs...", expanded=True) as status:
            try:
                result = vector_db.process_pdfs(
                    st.session_state.uploaded_pdf_paths if st.session_state.uploaded_pdf_paths else None
                )
                if result is not None:
                    st.session_state.vector_db_status = True
                    status.update(label="PDFs vectorized successfully!", state="complete")
                    st.success(f"✅ Successfully vectorized PDFs!")
                else:
                    status.update(label="No PDFs to vectorize!", state="error")
            except Exception as e:
                st.error(f"Error during vectorization: {str(e)}")
                status.update(label="Vectorization failed!", state="error")

    st.markdown("---")
    
    # Display vectorization status
    if vector_db.is_db_loaded():
        st.success("✅ Vector DB Ready")
        pdf_count = len(vector_db.get_pdf_files())
        if pdf_count > 0:
            st.info(f"📚 {pdf_count} PDF(s) in database")
    else:
        st.warning("⚠️ Please vectorize PDFs")

    st.markdown("---")

    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # Display chat statistics
    if st.session_state.chat_history:
        msg_count = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
        st.markdown(f"<p style='color: #ECECF1; font-size: 0.9rem;'>💬 Messages: {msg_count}</p>", unsafe_allow_html=True)


#--------------------------------------------------------------
# # CHAT INTERFACE
for message in st.session_state.chat_history:
    with st.container():
        if message['role'] == 'assistant':
            # Keep the assistant content as-is (math delimiters preserved)
            processed_content = process_latex(message['content'])
            formatted_content = format_message_content(processed_content)
        else:
            # Keep user messages as is (escaped for safety)
            formatted_content = html.escape(message['content'])

        st.markdown(f"""
            <div class="chat-message {message['role']}-message">
                <div class="message-content">
                    {formatted_content}
                </div>
            </div>
        """, unsafe_allow_html=True)
        # Trigger MathJax typesetting after each assistant message
        st.markdown(
            """
            <script>
            if (window.MathJax) {
                MathJax.Hub.Queue(['Typeset', MathJax.Hub]);
            }
            </script>
            """,
            unsafe_allow_html=True
        )

#----------------------------------------------------------------
# QUERY INPUT & PROCESSING
user_query = st.chat_input("Ask any query...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    with st.status("Processing query...", expanded=True) as status:
        try:
            # Step 1: Decide if retrieval is needed
            status.update(label="🤔 Analyzing query...", state="running")
            decision = decide_retrieval(user_query, llm)
            
            if decision == 'RETRIEVE':
                # Check if database is loaded
                if not vector_db.is_db_loaded():
                    st.error("Documents retrieval needed but no PDFs are vectorized. Please vectorize PDFs first.")
                    status.update(label="Error: No vector database", state="error")
                    st.session_state.chat_history.pop()
                else:
                    # Step 2: Retrieve documents
                    status.update(label="📚 Retrieving relevant documents...", state="running")
                    retrieved_docs = retrieve_docs(user_query)
                    
                    if not retrieved_docs:
                        st.warning("No relevant documents found. Answering without document context.")
                        status.update(label="⚠️ No documents found, answering directly...", state="running")
                        previous_history = st.session_state.chat_history[:-1]
                        response = answer_query_direct(llm, user_query, previous_history)
                    else:
                        # Step 3: Generate response with documents
                        status.update(label="✍️ Generating response with documents...", state="running")
                        previous_history = st.session_state.chat_history[:-1]
                        response = answer_query_with_docs(retrieved_docs, llm, user_query, previous_history)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    status.update(label="✅ Response generated with retrieval!", state="complete")
            
            else:  # DIRECT
                # Answer directly without retrieval
                status.update(label="💬 Generating direct response...", state="running")
                previous_history = st.session_state.chat_history[:-1]
                response = answer_query_direct(llm, user_query, previous_history)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                status.update(label="✅ Response generated!", state="complete")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            status.update(label="Error generating response", state="error")
            # Remove the user query from history since we couldn't process it
            if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'user':
                st.session_state.chat_history.pop()

    st.rerun()