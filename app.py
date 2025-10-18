import os
from dotenv import load_dotenv
import tempfile
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ----------------------------
# Load .env and set API key
# ----------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# ----------------------------
# LLM & embeddings
# ----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Streamlit page setup
# ----------------------------
st.set_page_config(page_title="LangChain Gemini Chatbot + PDF RAG", page_icon="🤖", layout="wide")
st.title("🤖 Gemini LangChain Chatbot + PDF RAG")

# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Place PDF uploader in the sidebar for a cleaner main interface
with st.sidebar:
    st.header("PDF Controls")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear PDF data"):
        st.session_state.vector_store = None
        st.sidebar.success("Vector store cleared for this session.")
        st.rerun()


# Process PDF if a new one is uploaded 
# This logic is kept separate from the chat flow
if uploaded_file is not None:
    # Use a flag in session state to avoid reprocessing the same file
    if st.session_state.get("processed_file_name") != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                split_docs = text_splitter.split_documents(docs)
                st.session_state.vector_store = FAISS.from_documents(split_docs, embedding_model)
                st.session_state.processed_file_name = uploaded_file.name
                st.success(f"✅ PDF '{uploaded_file.name}' processed and ready!")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
            finally:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)


# Display existing chat messages
# This now happens before the input box is defined
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)


# Chat input is now the last element, which pins it to the bottom
if user_input := st.chat_input("Type your message here..."):
    # Add user message to state and display it
    human_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(human_msg)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # This is your original logic, untouched
            if st.session_state.vector_store is not None:
                try:
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever,
                        chain_type="stuff"
                    )
                    bot_reply = qa_chain.run(user_input)
                except Exception as e:
                    bot_reply = f"Error during RAG: {e}"
            else:
                # Normal Gemini chat
                response = llm.invoke(st.session_state.messages)
                bot_reply = getattr(response, "content", str(response))

            # Add bot reply to state and display it
            assistant_msg = AIMessage(content=bot_reply)
            st.session_state.messages.append(assistant_msg)
            st.markdown(bot_reply)