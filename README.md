# project_codebasics_q_and_a

📊 CSV Q&A Assistant
This project is a web application built with Streamlit that functions as an intelligent assistant for analyzing CSV files. It leverages the power of LangChain, Cohere Embeddings, and Google Gemini to allow users to ask natural language questions about their data. The application processes a CSV file, creates a searchable knowledge base, and provides AI-powered answers based on the content of the file.

✨ Features
Dual API Support: Uses both Google Gemini for answering questions and Cohere for generating document embeddings.

Natural Language Queries: Ask complex questions about your data in plain English without needing to write any code.

Data Analysis & Preview: Get key metrics about your dataset (number of rows/columns) and a data preview directly in the app.

Persistent Chat History: The application keeps track of your Q&A session for easy reference.

Secure API Key Handling: API keys are entered via a password-protected text input for enhanced security.

Custom UI: A clean, organized, and responsive user interface built with Streamlit with custom CSS.

🚀 Getting Started
Prerequisites
You will need Python installed on your system. This project also requires API keys from both Google AI Studio and Cohere.

Google Gemini API Key: You can get a free key from Google AI Studio.

Cohere API Key: You can get a free key from the Cohere Dashboard.

Installation
Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Create and activate a virtual environment (recommended):

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

Install the required packages:

pip install -r requirements.txt

Note: The requirements.txt file should contain the following packages:

streamlit
pandas
langchain
langchain-cohere
langchain-google-genai
faiss-cpu # or faiss-gpu

🤖 Usage
Run the Streamlit application:

streamlit run main.py

Enter your API Keys: The application will prompt you to enter your API keys for both Gemini and Cohere in the sidebar.

Upload Your CSV: Use the file uploader to select and upload a .csv file. The application will automatically process the data and prepare it for analysis.

Ask Questions: Once the file is processed, a chat interface will appear. You can now type your questions in natural language, such as:

"How many rows and columns are there?"

"What are the data types of the different columns?"

"Show me the average value of the [specific column]."

"Find all records where [condition] is true."

View History: Your questions and the AI's answers will be displayed in the main window, allowing you to review the entire conversation.

🤝 Credits
This project was built using the following technologies:

Streamlit: For creating the interactive web application.

LangChain: For orchestrating the calls to the language models and handling data processing.

Google Gemini: The powerful large language model providing the core AI capabilities for answering questions.

Cohere: The embedding model used to convert text data into a searchable vector store.

FAISS: For efficient similarity search and retrieval of information from the processed data.
