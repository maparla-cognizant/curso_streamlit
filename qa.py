"""
Descripción: Este programa muestra como hacer un chatbot con Streamlit
Ejecución: streamlit run local_qa.py

Creado por: Mario Parreño
Fecha: 05-09-2023

Referencias:
    - https://streamlit.io/generative-ai
    - https://docs.streamlit.io/library/api-reference/chat/st.chat_message
    - https://docs.streamlit.io/library/api-reference/chat/st.chat_input
"""
import os
import uuid
import tempfile
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

from langchain import OpenAI
from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


#################################################################
################## CONFIGURACIÓN INICIAL ########################
#################################################################

# we split the documents into smaller chunks
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10,
    length_function=len,
)

MODEL_EMBEDDINGS = SentenceTransformer('all-MiniLM-L6-v2')

DB_CLIENT = chromadb.Client()
DB_COLLECTION = DB_CLIENT.get_or_create_collection(name="streamlit_course")

PROMPT_SKELETON = """
You are an exceptional reader that gently answer questions.

You know the following context information.

{chunks_formatted}

Answer to the following question from a customer.
Use only information from the previous context information.
Do not invent stuff.

Question: {query}

Answer:
"""

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["chunks_formatted", "query"],
    template=PROMPT_SKELETON,
)

LLM = OpenAI(model="text-davinci-003", temperature=0)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]


#################################################################
##################### TRATADO DE DATOS ##########################
#################################################################

uploaded_file = st.file_uploader("Elige un archivo PDF", type="pdf")
if uploaded_file is None:
    st.text("Por favor, sube un archivo PDF")
    st.stop()

pid = str(uuid.uuid4())

with st.spinner("Cargando y spliteando el documento..."):
    # Cargamos el archivo PDF subido por el usuario y lo spliteamos
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_filepath)
    doc = loader.load()
    chunks = TEXT_SPLITTER.split_documents(doc)
    st.success("Documento cargado y spliteado correctamente")

# Añadimos los chunks generados a nuestra db Deep Lake
with st.spinner("Añadiendo documentos a la base de datos..."):
    for chunk in chunks:
        chunk.metadata['pid'] = pid
        chunk_embedded = MODEL_EMBEDDINGS.encode(chunk.page_content)
        
        DB_COLLECTION.add(
            documents=[chunk.page_content],
            embeddings=[chunk_embedded.tolist()],
            metadatas=[chunk.metadata],
            ids=[str(uuid.uuid4())]
        )
    st.success("Documentos añadidos a ChromaDB!")
    # mostramos la cantidad de documentos en la db
    st.info(f"La base de datos tiene {DB_COLLECTION.count()} documentos")


#################################################################
##################### CHAT CON USUARIO ##########################
#################################################################

# Preguntamos al usuario
st.title("💬 Chatbot")
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# walrus operator: asigna y evalua a la vez una expresión
if query := st.chat_input():
    
    st.chat_message("user").write(query)

    query_embedded = MODEL_EMBEDDINGS.encode(query)
    similar_chunks = DB_COLLECTION.query(
        query_embeddings=[query_embedded.tolist()],
        n_results=3,
        where={"pid": pid}
    )
    retrieved_chunks = [chunk for chunk in similar_chunks["documents"][0]]

    # format the prompt
    chunks_formatted = "\n\n".join(retrieved_chunks)
    prompt_formatted = PROMPT_TEMPLATE.format(
        chunks_formatted=chunks_formatted,
        query=query
    )

    # generate answer
    respuesta = str(LLM(prompt_formatted))
    st.chat_message("assistant").write(respuesta)

    # guardamos la conversación
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
    