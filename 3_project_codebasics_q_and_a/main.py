import streamlit as st
import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import GoogleGenerativeAI
from langchain.schema import Document
import tempfile
import io

# Set page configuration
st.set_page_config(
    page_title="CSV Q&A Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #1e3d59;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .question-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: #f1f8e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin-bottom: 1rem;
    }
    .sidebar-info {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ff9800;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None

def setup_apis():
    """Setup API keys and validate them"""
    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### 🔑 API Configuration")
        
        gemini_api_key = st.text_input(
            "Google Gemini API Key", 
            type="password",
            help="Get your API key from Google AI Studio"
        )
        
        cohere_api_key = st.text_input(
            "Cohere API Key", 
            type="password",
            help="Get your API key from Cohere Dashboard"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔗 Get API Keys"):
            st.markdown("**Get your API keys from:**")
            st.markdown("- [Google AI Studio](https://makersuite.google.com/app/apikey)")
            st.markdown("- [Cohere Dashboard](https://dashboard.cohere.ai/api-keys)")
    
    return gemini_api_key, cohere_api_key

def validate_api_keys(gemini_key, cohere_key):
    """Validate that API keys are provided"""
    if not gemini_key or not cohere_key:
        st.error("⚠️ Please provide both Gemini and Cohere API keys to continue.")
        return False
    return True

def process_csv_file(uploaded_file):
    """Process the uploaded CSV file and create documents"""
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display basic info about the dataset
        st.success(f"✅ File uploaded successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Rows", len(df))
        with col2:
            st.metric("📋 Columns", len(df.columns))
        with col3:
            st.metric("💾 Size", f"{uploaded_file.size} bytes")
        
        # Show preview of the data
        with st.expander("👀 Preview Data (First 10 rows)"):
            st.dataframe(df.head(10))
        
        with st.expander("📈 Dataset Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Column Names:**")
                for col in df.columns:
                    st.write(f"- {col}")
            with col2:
                st.write("**Data Types:**")
                for col, dtype in df.dtypes.items():
                    st.write(f"- {col}: {dtype}")
        
        # Convert DataFrame to documents
        documents = []
        
        # Create documents from each row
        for index, row in df.iterrows():
            # Convert row to text format
            row_text = []
            for col, value in row.items():
                if pd.notna(value):  # Skip NaN values
                    row_text.append(f"{col}: {value}")
            
            content = " | ".join(row_text)
            documents.append(Document(
                page_content=content,
                metadata={"row_index": index, "source": uploaded_file.name}
            ))
        
        # Also create a summary document with column information
        summary_content = f"Dataset: {uploaded_file.name}\n"
        summary_content += f"Number of rows: {len(df)}\n"
        summary_content += f"Number of columns: {len(df.columns)}\n"
        summary_content += f"Columns: {', '.join(df.columns.tolist())}\n"
        
        # Add summary statistics for numerical columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            summary_content += f"Numerical columns: {', '.join(numeric_cols)}\n"
        
        documents.append(Document(
            page_content=summary_content,
            metadata={"type": "summary", "source": uploaded_file.name}
        ))
        
        return documents
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return None

def create_vector_store(documents, cohere_api_key):
    """Create vector store from documents"""
    try:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Split documents
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-english-light-v2.0"
        )
        
        # Create vector store
        vector_store = FAISS.from_documents(splits, embeddings)
        
        return vector_store
        
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return None

def setup_qa_chain(vector_store, gemini_api_key):
    """Setup the QA chain"""
    try:
        # Initialize Gemini LLM
        llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            return_source_documents=True
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"❌ Error setting up QA chain: {str(e)}")
        return None

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📊 CSV Q&A Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Upload your CSV file and ask questions about your data using AI!")
    
    # Setup APIs
    gemini_api_key, cohere_api_key = setup_apis()
    
    # Validate API keys
    if not validate_api_keys(gemini_api_key, cohere_api_key):
        st.stop()
    
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Your CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file to start asking questions about your data"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Check if this is a new file
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.chat_history = []  # Clear chat history for new file
            
            with st.spinner("🔄 Processing your file..."):
                # Process the CSV file
                documents = process_csv_file(uploaded_file)
                
                if documents:
                    # Create vector store
                    with st.spinner("🧠 Creating embeddings..."):
                        vector_store = create_vector_store(documents, cohere_api_key)
                    
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        
                        # Setup QA chain
                        with st.spinner("⚙️ Setting up AI assistant..."):
                            qa_chain = setup_qa_chain(vector_store, gemini_api_key)
                        
                        if qa_chain:
                            st.session_state.qa_chain = qa_chain
                            st.success("🎉 Your CSV file is ready for questions!")
        
        # Chat interface
        if st.session_state.qa_chain:
            st.markdown("### 💬 Ask Questions About Your Data")
            
            # Display chat history
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.markdown(f'<div class="question-box"><strong>❓ Question {i+1}:</strong> {question}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box"><strong>🤖 Answer:</strong> {answer}</div>', unsafe_allow_html=True)
            
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., How many records are in the dataset? What are the column names? Show me statistics about...",
                key="question_input"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("🚀 Ask Question", type="primary")
            with col2:
                clear_button = st.button("🗑️ Clear History")
            
            if clear_button:
                st.session_state.chat_history = []
                st.experimental_rerun()
            
            if ask_button and question:
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Get answer from QA chain
                        result = st.session_state.qa_chain({"query": question})
                        answer = result["result"]
                        
                        # Add to chat history
                        st.session_state.chat_history.append((question, answer))
                        
                        # Clear the input
                        st.session_state.question_input = ""
                        
                        # Rerun to show the new Q&A
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error getting answer: {str(e)}")
            
            # Sample questions
            with st.expander("💡 Sample Questions You Can Ask"):
                st.markdown("""
                - What is the structure of this dataset?
                - How many rows and columns are in the data?
                - What are the column names?
                - Can you summarize the data?
                - What are the data types of different columns?
                - Show me statistics about [specific column]
                - Find records where [condition]
                - What insights can you provide about this data?
                """)
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Add API Keys**: Enter your Gemini and Cohere API keys
        2. **Upload CSV**: Upload your CSV file
        3. **Ask Questions**: Ask questions about your data
        4. **Get Insights**: Receive AI-powered answers
        """)
        
        st.markdown("### ⚡ Features")
        st.markdown("""
        - **Smart Analysis**: AI understands your data structure
        - **Natural Language**: Ask questions in plain English  
        - **Multiple APIs**: Uses Gemini 1.5 Flash + Cohere
        - **Chat History**: Keep track of your Q&A session
        - **Data Preview**: See your data before asking questions
        """)
        
        st.markdown("### 🛠️ Supported File Types")
        st.markdown("- CSV files (.csv)")
        
        if st.session_state.uploaded_file_name:
            st.markdown("### 📊 Current Dataset")
            st.info(f"**File:** {st.session_state.uploaded_file_name}")

if __name__ == "__main__":
    main()