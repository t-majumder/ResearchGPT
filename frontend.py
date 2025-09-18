import streamlit as st
import os
import re
import base64
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Research GPT",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add MathJax support
st.markdown("""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
                inlineMath: [['$','$'], ['\\(','\\)']],
                displayMath: [['$$','$$'], ['\\[','\\]']],
                processEscapes: true
            }
        });
    </script>
""", unsafe_allow_html=True)

# Initialize LLMs
llm_llama = ChatGroq(
    model='llama-3.3-70b-specdec',
    api_key=os.getenv('GROQ_API_KEY')
)
llm_deepseek = ChatGroq(
    model='deepseek-r1-distill-llama-70b',
    api_key=os.getenv('GROQ_API_KEY')
)

class VectorDB:
    def __init__(self, pdfs_directory='pdfs', db_path='vectorstore'):
        self.pdfs_directory = pdfs_directory
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.faiss_db = None

    def process_pdfs(self):
        documents = self.load_pdfs()
        text_chunks = self.create_chunks(documents)
        self.faiss_db = FAISS.from_documents(text_chunks, self.embeddings)
        os.makedirs(self.db_path, exist_ok=True)
        self.faiss_db.save_local(self.db_path)
        return self.faiss_db

    def load_pdfs(self):
        documents = []
        for filename in os.listdir(self.pdfs_directory):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.pdfs_directory, filename)
                loader = PDFPlumberLoader(file_path)
                doc = loader.load()
                documents.extend(doc)
        return documents

    def create_chunks(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        return text_splitter.split_documents(documents)

    def get_retriever(self):
        if not self.faiss_db:
            if os.path.exists(os.path.join(self.db_path, 'index.faiss')):
                self.faiss_db = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
            else:
                self.faiss_db = self.process_pdfs()
        return self.faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 20})

def get_base64_of_background_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def retrieve_docs(query):
    retriever = vector_db.get_retriever()
    return retriever.invoke(query)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

def format_message_content(content):
    """Format message content with enhanced markdown and math support"""
    # Process math equations
    content = content.replace('$$', '<div class="math">', 1)
    if '$$' in content:
        content = content.replace('$$', '</div>', 1)
    
    # Process inline math
    content = re.sub(r'\$([^$]+)\$', r'<span class="inline-math">\1</span>', content)
    
    # Process markdown
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
    content = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', content, flags=re.DOTALL)
    content = re.sub(r'`(.*?)`', r'<code>\1</code>', content)
    
    # Bullet points
    content = re.sub(r'(?m)^- (.*)', r'<li>\1</li>', content)
    if '<li>' in content:
        content = '<ul>' + content + '</ul>'
    
    return content

def process_latex(text):
    """Process LaTeX in the text to make it render properly"""
    # Replace display math
    text = re.sub(r'\$\$(.*?)\$\$', r'<div class="math-container">\[\1\]</div>', text, flags=re.DOTALL)
    
    # Replace inline math
    text = re.sub(r'\$([^$]+?)\$', r'<span class="inline-math">\(\1\)</span>', text)
    
    # Process bullet points with LaTeX
    bullet_pattern = r'(\* .+?(?:\n|$))'
    def replace_bullet_with_latex(match):
        bullet_content = match.group(1)
        if '\\(' in bullet_content or '\\[' in bullet_content:
            return f'<div class="bullet-glass-container">{bullet_content}</div>'
        return bullet_content
    
    text = re.sub(bullet_pattern, replace_bullet_with_latex, text)
    
    return text

prompt_template = '''
You are Helpful assistant DoctorGPT. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.
Context: {context}
Question: {question}
Answer: '''

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context}).content

# Get background image
background_image_path = "Theme.jpg"
background_image = get_base64_of_background_image(background_image_path)

# Custom CSS with combined styles and math support
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{background_image}");
        background-size: cover;
    }}
    .glass-container {{
        background: rgba(255, 255, 255, 0.1);
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
        background: rgba(255, 255, 255, 0.1);
        max-width: 80%;
    }}
    .user-message {{
        background: rgba(0, 120, 212, 0.3);
        margin-left: auto;
        text-align: right;
    }}
    .assistant-message {{
        background: rgba(45, 45, 45, 0.4);
        margin-right: auto;
        text-align: left;
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
    .message-content {{
        color: #FFFFFF;
    }}
    .sidebar .block-container {{
        background: rgba(45, 45, 45, 0.7);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 10px;
    }}
    section[data-testid="stSidebar"] {{
        background: rgba(0, 0, 0, 0.3);
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
    }}
    .has-jax {{
        font-size: 100%;
    }}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_db_status' not in st.session_state:
    st.session_state.vector_db_status = False

# Initialize vector database
vector_db = VectorDB()

# Display header
st.markdown(
    """
    # 🔬 Research GPT - AI-Powered Paper Summarizer 🤖  
    **Your smart assistant for understanding and summarizing research papers!**  
    
    ---
    **✨ Features:**
    - 📄 Automatically processes PDFs from the 'pdfs' folder  
    - 🧠 Retrieve key insights using advanced AI  
    - 📚 Summarize complex concepts easily  
    - 🏆 Powered by cutting-edge Large Language Models (LLMs)  
    """
)

# Sidebar
with st.sidebar:
    st.image("Theam.jpg", use_container_width=True)
    st.markdown("<h1 style='color: #ECECF1;'>🔬 Research GPT</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("🔄 Vectorize PDFs"):
        with st.status("Processing PDFs...", expanded=True) as status:
            vector_db.process_pdfs()
            st.session_state.vector_db_status = True
            status.update(label="PDFs vectorized successfully!", state="complete")
    
    st.markdown("<h3 style='color: #ECECF1;'>🤖 Select the Model</h3>", unsafe_allow_html=True)
    llm_choice = st.selectbox(
        "Choose Language Model",
        ["Llama 3 LLM", "DeepSeek LLM"],
        help="Select the AI model for analysis"
    )
    llm = llm_llama if llm_choice == "Llama 3 LLM" else llm_deepseek
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Display PDF status
if os.path.exists(os.path.join(vector_db.db_path, 'index.faiss')):
    st.success("📚 Vector database is ready to use!")
else:
    st.warning("⚠️ Please click 'Vectorize PDFs' to process the documents in the 'pdfs' folder.")

# Chat interface
for message in st.session_state.chat_history:
    with st.container():
        if message['role'] == 'assistant':
            # Process LaTeX and formatting for assistant messages
            processed_content = process_latex(message['content'])
            formatted_content = format_message_content(processed_content)
        else:
            # Keep user messages as is
            formatted_content = message['content']
            
        st.markdown(f"""
            <div class="chat-message {message['role']}-message">
                <div class="message-content">
                    {formatted_content}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Ensure MathJax processes the new content
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

# Query input
user_query = st.chat_input("Ask about the research papers...")

if user_query:
    if os.path.exists(os.path.join(vector_db.db_path, 'index.faiss')):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.status("Generating response...", expanded=True) as status:
            try:
                # Retrieve relevant documents
                retrieved_docs = retrieve_docs(user_query)
                
                # Generate response
                response = answer_query(documents=retrieved_docs, model=llm, query=user_query)
                
                # Process LaTeX delimiters more robustly
                response = re.sub(r'\$\$(.*?)\$\$', r'\\[\1\\]', response, flags=re.DOTALL)
                response = re.sub(r'\$([^$]+?)\$', r'\\(\1\\)', response)
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                status.update(label="Response ready!", state="complete")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                status.update(label="Error generating response", state="error")
        
        st.rerun()
    else:
        st.error("Please vectorize the PDFs first using the 'Vectorize PDFs' button in the sidebar.")

# Footer with version info and additional links
st.markdown("""
    <div style='text-align: center; padding: 1rem; margin-top: 2rem; background: rgba(255, 255, 255, 0.1); border-radius: 10px;'>
        <p style='color: #ECECF1; margin-bottom: 0.5rem;'>Research GPT v1.0</p>
        <p style='color: #565869; font-size: 0.8rem;'>
            Built with 🔬 for Scientific Research<br>
            Powered by Groq LLMs
        </p>
    </div>
""", unsafe_allow_html=True)
