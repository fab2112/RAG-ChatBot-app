# Process Files
import os
import time
import json
import settings
import chromadb
import pandas as pd

from datetime import datetime
from typing import List, Tuple
from dotenv import load_dotenv
from urllib.parse import urlparse
from qdrant_client import QdrantClient
from utils.models import get_embeddings_model
from pinecone import Pinecone, ServerlessSpec
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders.csv_loader import CSVLoader
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader, UnstructuredCSVLoader, Docx2txtLoader

# Load .env
load_dotenv(dotenv_path=".env")


def save_uploaded_filenames(files: List[str], chunk_ids: List[List[str]]):
    """
    This function takes a list of gradio temporary file paths, extracts relevant information from each file (name,
    size, file type, upload timestamp and chunks ids), and stores this information in a JSON file called "uploaded_files.json".

    Parameters:
        files (List[str]): List of paths of uploaded files.
        chunk_ids (List[List[str]]): List of chunk-ids list of uploaded files.
    """

    json_path = "uploaded_files.json"

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    if files is not None and chunk_ids is not None:
        new_entries = []
        for file, chunks in zip(files, chunk_ids):
            file_name = os.path.basename(file.name)
            file_name_without_ext = os.path.splitext(file_name)[0]
            file_size = os.path.getsize(file.name) / 1024
            file_type = os.path.splitext(file_name)[1][1:].lower()
            timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

            entry = {
                "Time": timestamp,
                "File": file_name_without_ext,
                "Size": f"{file_size:.2f}kb",
                "Type": file_type,
                "Chunk-IDs": chunks  
            }
            new_entries.append(entry)

        existing_filenames = {entry["File"] for entry in existing_data}
        
        for entry in new_entries:
            existing_data.append(entry)
            existing_filenames.add(entry["File"])

        with open(json_path, "w") as f:
            json.dump(existing_data, f, indent=4)


def load_json_to_dataframe() -> pd.DataFrame:
    """
    This function reads the contents of the file "uploaded_files.json" and converts it to a pandas DataFrame.
    If the JSON file does not exist, it returns an empty DataFrame with the default columns: "Time", "File", "Size", and "Type".

    Returns:
        pandas.DataFrame: A DataFrame containing the uploaded files data, or an empty DataFrame if the JSON does not exist.
    """
    json_path = "uploaded_files.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    return pd.DataFrame(columns=["Time", "File", "Size", "Type", "Chunk-IDs"])


def process_textsplitter(
    docs: List, txt_splitter: str, chk_size: int, chk_overlap: int
) -> List:
    """
    Processes documents and splits them into smaller parts using different text splitting strategies.

    This function applies a selected text splitting method to a collection of documents,
    returning the documents split into chunks according to the chosen method.

    Parameters:
        docs: List of documents to be split.
        txt_splitter (str): Name of the text splitting method to be used.
        chk_size (int): Maximum size (in characters) of each chunk for character-based splitters.
        chk_overlap (int): Number of overlapping characters between chunks, used only for character-based splitters.

    Returns:
        List: List of documents split according to the chosen method.
    """

    if txt_splitter == "RecursiveCharacterTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chk_size,
            chunk_overlap=chk_overlap,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(docs)

    elif txt_splitter == "CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(
            chunk_size=chk_size,
            chunk_overlap=chk_overlap,
            separator="\n\n",
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(docs)

    elif txt_splitter == "SemanticChunker":
        embeddings_model = get_embeddings_model()[0]
        threshold_type = (
            "percentile"  # "standard_deviation" | "interquartile | percentile"
        )
        text_splitter = SemanticChunker(
            embeddings=embeddings_model, breakpoint_threshold_type=threshold_type
        )
        all_splits = text_splitter.split_documents(docs)

    return all_splits

def process_PDFloader(file: str, pdf_loader: str) -> List:
    """
    Loads a PDF file and extracts its contents using the specified loading method.

    This function uses one of the available loaders to read and process the contents of a PDF file,
    returning the extracted contents as a list of documents.

    Parameters:
        file (str): Path to the PDF file to be loaded - 'gradio.utils.NamedString'.
        pdf_loader (str): Name of the PDF loader to be used.

    Returns:
        List: List of documents extracted from the PDF.
    """

    if pdf_loader == "PyPDFLoader":
        loader = PyPDFLoader(file)
        docs = loader.load()

    elif pdf_loader == "PDFPlumberLoader":
        loader = PDFPlumberLoader(file)
        docs = loader.load()

    return docs


def process_CSVloader(file: str, csv_loader: str) -> List:
    """
    Loads a CSV file and extracts its contents using the specified loading method.

    This function uses one of the available loaders to read and process the contents of a CSV file,
    returning the extracted contents as a list of documents.

    Parameters:
        file (str): Path to the CSV file to be loaded - 'gradio.utils.NamedString'.
        csv_loader (str): Name of the CSV loader to be used.

    Returns:
        List: List of documents extracted from the CSV.
    """

    if csv_loader == "CSVLoader":
        loader = CSVLoader(file)
        docs = loader.load()

    elif csv_loader == "UnstructuredCSVLoader":
        loader = UnstructuredCSVLoader(file)
        docs = loader.load()

    return docs


def process_MICROSOFTDOCSloader(file: str) -> List:
    """
    Loads a Microsof docs (word | excel) and extracts its contents using the specified loading method.

    Parameters:
        file (str): Path to the doc file to be loaded - 'gradio.utils.NamedString'.

    Returns:
        List: List of documents extracted.
    """

    if file.name.endswith((".xlsx", ".xls")):
        loader = UnstructuredExcelLoader(file)
    elif file.name.endswith(".docx"):
        loader = Docx2txtLoader(file)
    docs = loader.load()
    return docs


def handle_files_to_qdrant(
    files: List[str],
    txt_splitter: str,
    pdf_loader: str,
    chk_size: int,
    chk_overlap: int,
    csv_loader: str,
) -> str:
    """
    Processes uploaded files, generates embeddings of their contents and stores the vectors in Qdrant vector database.

    This function handles the processing of different types of files, divides the texts into chunks,
    generates embeddings for each chunk and inserts these vectors into the Qdrant vector database.
    It also saves and updates information about files already loaded into the database.

    Parameters:
        files (List[str]): List of path files uploaded to be processed.
        txt_splitter (str): Name of the text splitting strategy to be used.
        pdf_loader (str): Name of the PDF loader to be used.
        chk_size (int): Maximum size of chunks (in number of characters) for text splitting.
        chk_overlap (int): Number of overlapping characters between chunks.
        csv_loader (str): Name of the CSV loader to be used.

    Returns:
        str: Message indicating the number of chunks added to the Qdrant collection.

    Note:
    - Supported files: PDFs (.pdf), CSVs (.csv), and Microsoft Office documents (.xlsx, .xls, .docx).
    - Rate limit (`EMBEDDINGS_RATE_LIMIT`) is applied between embedding generations.
    - Information about uploaded files is recorded in the "uploaded_files.json" file.
    """
    
    # Set chunk_id_ref
    json_path = "uploaded_files.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            df = pd.DataFrame(json.load(f))
            last_row = df.iloc[-1]
            chunk_id_ref = last_row["Chunk-IDs"][-1]
    else:
        chunk_id_ref = 0

    # Load model embeddings
    embeddings = get_embeddings_model()
    embeddings_model = embeddings[0]
    vector_dim = embeddings[1]

    # Connect to Qdrant
    client = QdrantClient(url=settings.QDRANT_URL)
    collection_name = settings.DATABASE_NAME

    # Create collection if it doesn't exist
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    if not files:
        return "No files uploaded."

    points = []
    all_chunk_ids = []  
    total_chunks = 0

    for file in files:
        if file.name.endswith(".pdf"):
            docs = process_PDFloader(file, pdf_loader)
        elif file.name.endswith(".csv"):
            docs = process_CSVloader(file, csv_loader)
        elif file.name.endswith((".xlsx", ".xls", ".docx")):
            docs = process_MICROSOFTDOCSloader(file)
        else:
            continue  

        # Divide em chunks
        all_splits = process_textsplitter(docs, txt_splitter, chk_size, chk_overlap)

        chunk_ids = []
        for doc in all_splits:
            chunk_embedding = embeddings_model.embed_documents([doc.page_content])[0]

            payload = {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }

            chunk_id_ref += 1
            chunk_ids.append(chunk_id_ref)

            point = PointStruct(
                id=chunk_id_ref,
                vector=chunk_embedding,
                payload=payload
            )
            points.append(point)

            time.sleep(settings.EMBEDDINGS_RATELIMIT)

        all_chunk_ids.append(chunk_ids)

    # Insert points in Qdrant
    if points:
        client.upsert(collection_name=collection_name, points=points)

    # Save information about uploaded files
    save_uploaded_filenames(files, all_chunk_ids)
    
    total_chunks = sum(len(sublist) for sublist in all_chunk_ids)

    return f"{total_chunks} chunks added to Qdrant."

def handle_files_to_pinecone(
    files: List[str],
    txt_splitter: str,
    pdf_loader: str,
    chk_size: int,
    chk_overlap: int,
    csv_loader: str,
) -> str:
    """
    Processes uploaded files, generates embeddings of their contents and stores the vectors in Pinecone vector database.

    This function handles the processing of different types of files, divides the texts into chunks,
    generates embeddings for each chunk and inserts these vectors into the Pinecone vector database.
    It also saves and updates information about files already loaded into the database.

    Parameters:
        files (List[str]): List of path files uploaded to be processed.
        txt_splitter (str): Name of the text splitting strategy to be used.
        pdf_loader (str): Name of the PDF loader to be used.
        chk_size (int): Maximum size of chunks (in number of characters) for text splitting.
        chk_overlap (int): Number of overlapping characters between chunks.
        csv_loader (str): Name of the CSV loader to be used.

    Returns:
        str: Message indicating the number of chunks added to the Qdrant collection.

    Note:
    - Supported files: PDFs (.pdf), CSVs (.csv), and Microsoft Office documents (.xlsx, .xls, .docx).
    - Rate limit (`EMBEDDINGS_RATE_LIMIT`) is applied between embedding generations.
    - Information about uploaded files is recorded in the "uploaded_files.json" file.
    """
    
    # Set chunk_id_ref
    json_path = "uploaded_files.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            df = pd.DataFrame(json.load(f))
            last_row = df.iloc[-1]
            chunk_id_ref = last_row["Chunk-IDs"][-1]
    else:
        chunk_id_ref = 0

    # Load model embeddings
    embeddings = get_embeddings_model()
    embeddings_model = embeddings[0]
    vector_dim = embeddings[1]

    # Initialize the Pinecone client
    pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=settings.PINECONE_REGION)
    index_name = settings.DATABASE_NAME

    # Create index if not exists
    if index_name not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=index_name,
            dimension=vector_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION),
        )

    index = pinecone.Index(index_name)

    if not files:
        return "No files uploaded."
    
    all_chunk_ids = []  
    total_chunks = 0

    vectors = []
    for file in files:
        if file.name.endswith(".pdf"):
            docs = process_PDFloader(file, pdf_loader)
        elif file.name.endswith(".csv"):
            docs = process_CSVloader(file, csv_loader)
        elif file.name.endswith((".xlsx", ".xls", ".docx")):
            docs = process_MICROSOFTDOCSloader(file)
        else:
            continue  

        # Split into chunks
        all_splits = process_textsplitter(docs, txt_splitter, chk_size, chk_overlap)

        chunk_ids = []
        for i, doc in enumerate(all_splits):
            # Generate embedding for the chunk
            chunk_embedding = embeddings_model.embed_documents([doc.page_content])[0]

            # Build metadata
            metadata = {
                "content": doc.page_content,
            }
            metadata.update(doc.metadata)
            
            chunk_id_ref += 1
            chunk_ids.append(chunk_id_ref)

            # Create the vector for upsert
            vector = (
                str(chunk_id_ref),  # ID
                chunk_embedding,    # Vector
                metadata,           # Data + Metadata
            )
            vectors.append(vector)

            time.sleep(settings.EMBEDDINGS_RATELIMIT)
            
        all_chunk_ids.append(chunk_ids)

    # Add vectors to Pinecone
    if vectors:
        index.upsert(vectors=vectors)

    # Save names of uploaded files
    save_uploaded_filenames(files, all_chunk_ids)
    
    total_chunks = sum(len(sublist) for sublist in all_chunk_ids)

    return f"{total_chunks} chunks added to Pinecone."

def handle_files_to_chomadb(
    files: List[str],
    txt_splitter: str,
    pdf_loader: str,
    chk_size: int,
    chk_overlap: int,
    csv_loader: str,
) -> str:
    """
    Processes uploaded files, generates embeddings of their contents and stores the vectors in ChromaDb vector database.

    This function handles the processing of different types of files, divides the texts into chunks,
    generates embeddings for each chunk and inserts these vectors into the ChromaDb vector database.
    It also saves and updates information about files already loaded into the database.

    Parameters:
        files (List[str]): List of path files uploaded to be processed.
        txt_splitter (str): Name of the text splitting strategy to be used.
        pdf_loader (str): Name of the PDF loader to be used.
        chk_size (int): Maximum size of chunks (in number of characters) for text splitting.
        chk_overlap (int): Number of overlapping characters between chunks.
        csv_loader (str): Name of the CSV loader to be used.

    Returns:
        str: Message indicating the number of chunks added to the Qdrant collection.

    Note:
    - Supported files: PDFs (.pdf), CSVs (.csv), and Microsoft Office documents (.xlsx, .xls, .docx).
    - Rate limit (`EMBEDDINGS_RATE_LIMIT`) is applied between embedding generations.
    - Information about uploaded files is recorded in the "uploaded_files.json" file.
    """
    
    # Set chunk_id_ref
    json_path = "uploaded_files.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            df = pd.DataFrame(json.load(f))
            last_row = df.iloc[-1]
            chunk_id_ref = last_row["Chunk-IDs"][-1]
    else:
        chunk_id_ref = 0

    embeddings = get_embeddings_model()[0]

    # Configure the client - ChromaDB connection
    parsed = urlparse(settings.CHROMADB_URL)
    client = chromadb.HttpClient(host=parsed.hostname, port=int(parsed.port))

    # Create or get a collection in ChromaDB if it already exists
    collection_name = settings.DATABASE_NAME
    try:
        collection = client.get_collection(name=collection_name)
    except:
        collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    if not files:
        return "No files uploaded."
    
    all_chunk_ids = []  
    total_chunks = 0

    for file in files:
        if file.name.endswith(".pdf"):
            docs = process_PDFloader(file, pdf_loader)
        elif file.name.endswith(".csv"):
            docs = process_CSVloader(file, csv_loader)
        elif file.name.endswith((".xlsx", ".xls", ".docx")):
            docs = process_MICROSOFTDOCSloader(file)

        all_splits = process_textsplitter(docs, txt_splitter, chk_size, chk_overlap)

        # Prepare data for ChromaDb  
        chunk_ids = []
        documents = []
        metadatas = []
        chunk_embeddings = []
        for _, doc in enumerate(all_splits):
            chunk_id_ref += 1
            chunk_ids.append(chunk_id_ref)
            documents.append(doc.page_content)
            metadatas.append(doc.metadata)
            #
            # Generate chunk embeddings
            chunk_embeddings.extend(embeddings.embed_documents([doc.page_content]))
            #
            time.sleep(settings.EMBEDDINGS_RATELIMIT)

        # Add to ChromaDB
        collection.add(
            documents=documents,
            embeddings=chunk_embeddings,
            metadatas=metadatas,
            ids=list(map(str, chunk_ids)),
        )

        time.sleep(settings.EMBEDDINGS_RATELIMIT)
        
        all_chunk_ids.append(chunk_ids)
        
    # Save names of uploaded files
    save_uploaded_filenames(files, all_chunk_ids)
    
    total_chunks = sum(len(sublist) for sublist in all_chunk_ids)

    return f"{total_chunks} chunks added to ChromaDB."

def process_files(
    files: List[str],
    txt_splitter: str,
    pdf_loader: str,
    chk_size: int,
    chk_overlap: int,
    csv_loader: str,
) -> str:
    """
    Processes uploaded files and stores their embeddings in the specified vector database.

    This function determines which vector storage service to use (Qdrant, ChromaDB, or Pinecone)
    based on the configuration defined in `settings.VECSTORAGE`, and then directs the processing of the files
    to the corresponding function.

    Parameters:
        files (List[str]): List of path files uploaded to be processed.
        txt_splitter (str): Name of the text splitting strategy to be used.
        pdf_loader (str): Name of the PDF loader to be used.
        chk_size (int): Maximum size of chunks (in number of characters) for text splitting.
        chk_overlap (int): Number of overlapping characters between chunks.
        csv_loader (str): Name of the CSV loader to be used.

    Returns:
        str: Message reporting the result of the operation on the chosen vector service.
    """

    if settings.VECSTORAGE == "qdrant":
        return handle_files_to_qdrant(
            files, txt_splitter, pdf_loader, chk_size, chk_overlap, csv_loader
        )
    elif settings.VECSTORAGE == "chromadb":
        return handle_files_to_chomadb(
            files, txt_splitter, pdf_loader, chk_size, chk_overlap, csv_loader
        )
    elif settings.VECSTORAGE == "pinecone":
        return handle_files_to_pinecone(
            files, txt_splitter, pdf_loader, chk_size, chk_overlap, csv_loader
        )
