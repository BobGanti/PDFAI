import os
import openai
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
#from langchain.chains.question_answering import load_qa_chain
#from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(".env")
openai.api_key = os.getenv('OPENAI_API_KEY')

# Extracting the content of each pdf page, go through all uploaded pdf files
def get_pdf_text(pdfs):
    return "".join(page.extract_text() for pdf in pdfs for page in PdfReader(pdf).pages)

# Splitting the text into chunks of given word count and overlap
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=len
    )
    return splitter.split_text(text)

# Embed the chuncks and and format as faiss vectorstore.
def get_vectorstore(text_chunks):
    embedding_model = OpenAIEmbeddings()
    return FAISS.from_texts(text_chunks, embedding_model)

# The retriever model receives the vectorstore and returns the chain.
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(), 
        memory=memory)
    return conversational_chain

def main():
    st.set_page_config(page_title="PDF AI", page_icon=":books:")
    st.header("Query PDFs Assistant:books:")
    st.write(css, unsafe_allow_html=True)

    if 'processed' not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Submit'", accept_multiple_files=True)

        if st.button("Submit"):
            with st.spinner("Processing..."):
                pdf_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(pdf_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.processed = True
                st.success("Processing complete")

    user_question = st.chat_input("Ask about your PDFs...")
    if user_question:
        response = st.session_state.conversation(
            {'question':user_question})
        for i, message in enumerate(response["chat_history"]):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
