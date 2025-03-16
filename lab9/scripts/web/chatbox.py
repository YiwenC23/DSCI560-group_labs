import os
import faiss
import ollama
import getpass
import numpy as np
from openai import OpenAI
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter


#* Get the API key from the environment variable, ask for input if not found
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")
client = OpenAI()


llm = ["gpt-4o-mini", "gemma3:27b-it-q4_K_M" ]
emb_models = ["text-embedding-3-small", "nomic-embed-text"]


def extract_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text


def chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    return chunks


def embed_text(text_chunks, model):
    if model == "text-embedding-3-small":
        response = client.embeddings.create(
            input = text_chunks,
            model = model
        )
        embeddings = [item.embedding for item in response.data]
    elif model == "nomic-embed-text":
        embeddings = ollama.embed(
            model = model,
            input = text_chunks
        ).embeddings
    else:
        raise ValueError(f"Model {model} not supported")
    
    return embeddings


def create_vector_store(embeddings):
    embeddings_array = np.array(embeddings).astype("float32")
    vector_dim = embeddings_array.shape[1]
    vector_store = faiss.IndexFlatL2(vector_dim)
    vector_store.add(embeddings_array)
    
    return vector_store


def conversation_chain(query, text_chunks, vector_store, chat_history, model):
    #? # Append user question to chat history
    chat_history.append(f"User: {query}")
    
    #? # Compute embedding for the query
    query_embedding = embed_text([query], emb_models[1])[0]
    query_embedding_array = np.array([query_embedding]).astype("float32")
    
    #? # Retrieve context from the vector store
    k = 4
    _, indices = vector_store.search(query_embedding_array, k)
    context_chunks = "\n".join([text_chunks[i] for i in indices[0]])
    
    #? # Build prompt with chat history and retrieved context
    history_text = "\n".join(chat_history)
    prompt = f"{history_text}\nContext: {context_chunks}\nUser: {query}\nBot:"
    
    #? Get answer from the LLM
    
    if model == "gpt-4o-mini":
        ChatResponse = client.chat.completions.create(
            model = llm[0],
            messages = [
                {"role": "system", "content": "You are a helpful assistant that can answer questions about the context provided."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = ChatResponse.choices[0].message.content.strip()
    elif model == "gemma3:27b-it-q4_K_M":
        ChatResponse = ollama.chat(
            model = llm[1],
            messages = [
                {"role": "user", "content": prompt}
            ]
        )
        answer = ChatResponse.message.content.strip()
    else:
        raise ValueError(f"Model {model} not supported")
    
    #? Append bot answer to chat history
    chat_history.append(f"Bot: {answer}")
    
    return answer


# def driver():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with PDFs",
#                        page_icon=":robot_face:")
#     st.write(css, unsafe_allow_html=True)
    
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
#     if "vector_store" not in st.session_state:
#         st.session_state.vector_store = None
#     if "text_chunks" not in st.session_state:
#         st.session_state.text_chunks = None
    
#     st.header("Chat with PDFs :robot_face:")
    
#     #? Sidebar: Process PDF(s) and create vector store
#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
#         if st.button("Process") and pdf_docs:
#             with st.spinner("Processing"):
#                 raw_text = extract_pdf_text(pdf_docs)
#                 text_chunks = chunk_text(raw_text)
#                 embeddings = embed_text(text_chunks)
#                 vector_store = create_vector_store(embeddings)
#                 st.session_state.text_chunks = text_chunks
#                 st.session_state.vector_store = vector_store
#             st.success("PDF processed!")
    
#     #? # Main area: Chat interface
#     if st.session_state.vector_store and st.session_state.text_chunks:
#         user_question = st.text_input("Ask questions about your documents:")
#         if user_question:
#             answer = conversation_chain(
#                 user_question,
#                 st.session_state.text_chunks,
#                 st.session_state.vector_store,
#                 st.session_state.chat_history
#             )
            
#             #? Display conversation history
#             for message in st.session_state.chat_history:
#                 if message.startswith("User:"):
#                     st.write(user_template.replace("{{MSG}}", message[len("User: "):]), unsafe_allow_html=True)
#                 else:
#                     st.write(bot_template.replace("{{MSG}}", message[len("Bot: "):]), unsafe_allow_html=True)


# if __name__ == "__main__":
#     driver()