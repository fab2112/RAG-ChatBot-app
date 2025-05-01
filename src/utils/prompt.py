# Process Prompt
import os
import settings
import chromadb

from pinecone import Pinecone
from dotenv import load_dotenv
from urllib.parse import urlparse
from qdrant_client import QdrantClient
from utils.models import get_embeddings_model

# Load .env
load_dotenv(dotenv_path=".env")


def retrieves_qdrant(question: str, ctx_chunks: int) -> str:
    """
    Retrieves relevant chunks stored in Qdrant based on a given question.

    The function generates the question embedding, queries the "data_collection" collection in Qdrant,
    and returns a formatted context containing the most relevant documents based on similarity.

    Parameters:
        question (str): The question or query to be used to retrieve similar documents.
        ctx_chunks (int): The number of chunks to return as context.

    Returns (str): 
        A text containing the relevant content found in Qdrant plus metadata,
        If the collection is not found, returns an error message.
    """

    # Get embeddings model
    embeddings = get_embeddings_model()
    embeddings_model = embeddings[0]
    
    # Connect to Qdrant
    client = QdrantClient(url=settings.QDRANT_URL)
    collection_name = settings.DATABASE_NAME
    
    try:
        client.get_collection(collection_name)
    except:
        return "Collection not found on Qdrant."
    
    # Generate question embedding
    question_embedding = embeddings_model.embed_query(question)

    # Search for relevant documents
    search_results = client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=ctx_chunks,
        with_payload=True
    )
    
    # Set up the context
    context = ""
    for i, result in enumerate(search_results, start=1):
        content = result.payload.get("content", "").strip()
        metadata = result.payload.get("metadata", {})
        
        # Source filename
        source_path = metadata.get("source", "unknown")
        document_name = os.path.basename(source_path)

        # Page of chunk
        page = metadata.get("page", None)

        # Adjust page number 
        if page is not None:
            page_display = f"{page + 1}"  
        else:
            page_display = "unknown"

        if content:
            context += (
                f"CONTEXT_{i}:\n"
                f"Document name: {document_name}\n"
                f"Page: {page_display}\n"
                f"Content:\n{content}\n\n"
            )

    return context.strip()

  
def retrieves_pinecone(question: str, ctx_chunks: int) -> str:
    """
    Retrieves relevant chunks stored in Pinecone based on a given question.

    The function generates the question embedding, queries the "data-index" index in Pinecone
    and returns a formatted context containing the most relevant documents based on similarity.

    Parameters:
        question (str): The question or query to be used to retrieve similar documents.
        ctx_chunks (int): The number of chunks to return as context.

    Returns (str): 
        A text containing the relevant content found in Pinecone plus metadata,
        If the index is not found, returns an error message.
    """
 
    # Get embeddings model
    embeddings = get_embeddings_model()
    embeddings_model = embeddings[0]
    
    pinecone = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=settings.PINECONE_REGION
    )

    index_name = settings.DATABASE_NAME

    # Check if index exists
    if index_name not in pinecone.list_indexes().names():
        return "Index not found in Pinecone."

    index = pinecone.Index(index_name)

    # Generate embedding for the question
    question_embedding = embeddings_model.embed_query(question)

    # Search Pinecone
    search_results = index.query(
        vector=question_embedding,
        top_k=ctx_chunks,
        include_metadata=True
    )

    # Set up the context
    context = ""
    for i, match in enumerate(search_results.get('matches', []), start=1):
        metadata = match.get('metadata', {})
        
        content = metadata.get("content", "").strip()
        source_path = metadata.get("source", "unknown")
        document_name = os.path.basename(source_path)
        page = metadata.get("page", None)

        # Adjust page number
        if page is not None:
            page_display = f"{page + 1}"  
        else:
            page_display = "unknown"

        if content:
            context += (
                f"CONTEXT_{i}:\n"
                f"Document name: {document_name}\n"
                f"Page: {page_display}\n"
                f"Content:\n{content}\n\n"
            )

    return context.strip()


def retrieves_chromadb(question: str, ctx_chunks: int) -> str:
    """
    Retrieves relevant chunks stored in ChromaDb based on a given question.

    The function generates the question embedding, queries the "data-collection" collection in ChromaDb
    and returns a formatted context containing the most relevant documents based on similarity.

    Parameters:
        question (str): The question or query to be used to retrieve similar documents.
        ctx_chunks (int): The number of chunks to return as context.

    Returns (str): 
        A text containing the relevant content found in ChromaDb plus metadata,
        If the collection is not found, returns an error message.
    """

    # Get embeddings model
    embeddings = get_embeddings_model()
    embeddings_model = embeddings[0]
        
    # Connect to ChromaDB
    parsed = urlparse(settings.CHROMADB_URL)
    client = chromadb.HttpClient(host=parsed.hostname, port=int(parsed.port))
    collection_name = settings.DATABASE_NAME
    
    # Get collection
    try:
        collection = client.get_collection(name=collection_name)
    except:
        return "Collection not found in ChromaDB."
    
    # Generate question embedding
    question_embedding = embeddings_model.embed_query(question)
    
    # Search for the most relevant documents
    search_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=ctx_chunks,
        include=["documents", "metadatas"]
    )
    
    documents = search_results.get("documents", [[]])[0]
    metadatas = search_results.get("metadatas", [[]])[0]
    
    # Set up the context
    context = ""
    for i, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
        content = doc.strip()
        
        # Source filename
        source_path = metadata.get("source", "unknown")
        document_name = os.path.basename(source_path)

        # Page of chunk
        page = metadata.get("page", None)

        # Adjust page number 
        if page is not None:
            page_display = f"{page + 1}" 
        else:
            page_display = "unknown"

        if content:
            context += (
                f"CONTEXT_{i}:\n"
                f"Document name: {document_name}\n"
                f"Page: {page_display}\n"
                f"Content:\n{content}\n\n"
            )
    
    return context.strip()


def process_prompt(messages: str, question: str, rag_mode: str, ctx_chunks: int) -> str:
    """
    Constructs the final prompt that will be sent to the language model,
    with or without additional context information, depending on the rag_mode.

    Parameters:
        messages (str): Conversation history.
        question (str): Current question from the user.
        rag_mode (str): Retrieval mode ("ON" to include context via RAG, "OFF" to use only the question and history).
        ctx_chunks (int): Number of context chunks to retrieve if rag_mode is enabled.

    Returns (str):
        A structured prompt, containing the question, the conversation history,
        and optionally additional context retrieved from vector database according to rag_mode.
    """
    
    if rag_mode == "ON":
        
        if settings.RETRIEVER == "qdrant":
            context = retrieves_qdrant(question, ctx_chunks)
        elif settings.RETRIEVER == "pinecone":
            context = retrieves_pinecone(question, ctx_chunks)
        elif settings.RETRIEVER == "chromadb":
            context = retrieves_chromadb(question, ctx_chunks)
            
        prompt_main = f"""
        <question>
        {question}
        </question>

        <messages>
        {messages}
        </messages>

        <context>
        {context}
        </context>

        Answer the main question in <question> ONLY based on the context provided in <context> and the chat messages
        in <messages>. If you can't find information in the context, just say "I couldn't find that information in my
        database, please specify the question further" in language {settings.LANGUAGE}. Keep your answer as concise as
        possible according to the rule.
        
        Respond the question ONLY in {settings.LANGUAGE} language.
        
        """
    elif rag_mode == "OFF":
        
        prompt_main = f"""
        
            <question>
            {question}
            </question>

            <messages>
            {messages}
            </messages>
            
            Answer the main question in <question>. Use the conversation history in <messages> for better
            understanding.
            
            Respond the question ONLY in {settings.LANGUAGE} language.
            
            """
    
    return prompt_main
    