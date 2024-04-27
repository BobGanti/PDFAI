# PDF AI Assistant with FAISS VECTORSTORE
PDF AI Assistant is a conversational AI chatbot built using Streamlit, designed to query and understand content from specified PDF documents. By leveraging the power of LangChain, OpenAI's embeddings, and Streamlit's interactive capabilities, PDFAI Assistant offers an intuitive interface for users to interact with and retrieve information from a knowledge base constructed from document content.

# Features #
  a. PDF Document Processing: PDFAI Assistant allows for the upload of PDF documents, enhancing its knowledge base with the content extracted from these documents.
  b. Conversational Interface: Built with Streamlit, the application provides a chat-like interface for querying the knowledge base in a natural, conversational manner.
  c. Customizable Prompts: Integrates with OpenAI's embeddings for customizable prompt engineering, enabling refined responses based on the content of the knowledge base.

# Installation #
Before running SMAI Assistant, ensure you have Python 3.6 or later installed. You can then install the necessary dependencies via pip:
git clone https://github.com/BobGanti/PDFAI.git
cd PDFAI
pip install -r requirements.txt

# Dependencies #
  > Streamlit
  > LangChain
  > PyMuPDF (for PDF processing)
  > dotenv (for environment variable management)

# Setup #
1. Customsation: Modify the .env.local file to tailor the assistant's behavior and responses to your preferences.
  configuration:
  a. OPENAI_API_KEY: Your OpenAI API key for embeddings and chat.
  b. INSTRUCTIONS: Default instructions for querying.
  c. ASSISTANT_PROFILE: Customize the assistant's profile.
3. Run the Application: Start the SMAI Assistant by running the Streamlit application.<br>
  "streamlit run pdfs.py

# Usage #
## Adding Content
  > Use the sidebar to upload PDF documents to augment the chatbot's knowledge base.<br>
  > To Interact with the Chatbot, enter queries in the chat input field to receive responses based on the aggregated knowledge from the specified uploaded files.<br>

# Contributing #
Contributions are welcome! If you have suggestions for improvements or bug fixes, please open an issue or submit a pull request.

# License #
MIT License - See the LICENSE file for details.
