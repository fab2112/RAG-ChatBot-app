# Get Models
import os
import settings

from dotenv import load_dotenv
from typing import List, Tuple
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load .env
load_dotenv(dotenv_path=".env")

  
def get_model(llm_type: str, llm_temp: float, llm_top_p: float) -> object:
    """
    Returns a language model (LLM) instance configured according to the specified type.

    This function dynamically selects the LLM provider based on the `llm_type` parameter prefix
    and returns an already configured instance with the given temperature and top-p.

    Parameters:
        llm_type (str): Type and name of the model in the format "modelProvider_modelName".
        llm_temp (float): Temperature to be used by the model to control the randomness of the generation.
        llm_top_p (float): Top-p value for token sampling.

    Returns:
        Object model: Configured language model instance.
    """
    
    # Google
    if llm_type.startswith("Google"):
        model = llm_type.split("_", 1)[1]
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=llm_temp,
            top_p=llm_top_p,
    )
        
    # Groq
    elif llm_type.startswith("Groq"):
        model = llm_type.split("_", 1)[1]
        return ChatGroq(
            model_name=model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=llm_temp,
            model_kwargs={"top_p": llm_top_p}
    )
    
    # OpenAI
    if llm_type.startswith("OpenAI"):
        model = llm_type.split("_", 1)[1]
        return ChatOpenAI(
            model_name=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=llm_temp,
            top_p=llm_top_p,
        )
    
    # OpenAI
    if llm_type.startswith("Nvidia"):
        model = llm_type.split("_", 1)[1]
        return ChatNVIDIA(
            model_name=model,
            openai_api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=llm_temp,
            top_p=llm_top_p,
        )
        
    # Ollama
    elif llm_type.startswith("Ollama"):
        model = llm_type.split("_", 1)[1]     
        return ChatOllama(
            model=model,
            temperature=llm_temp,
            base_url=settings.OLLAMA_URL,
            model_kwargs={"top_p": llm_top_p}
    )


def get_embeddings_model() -> Tuple[object, int]:
    """
    Returns the configured embedding model, along with its vector dimension.

    The function dynamically selects the embedding provider based on the configuration defined in
    `settings.EMBEDDINGS_MODEL` and returns the model instance with the dense vector dimension.

    Returns:
        Tuple[object model, int]:
    """
    
    key = list(settings.EMBEDDINGS_MODEL.keys())[0]
     
    if key == "Pinecone":
        embeddings_model = PineconeEmbeddings(
        model=settings.EMBEDDINGS_MODEL[key][0],
        api_key=os.getenv("PINECONE_API_KEY")
    )
    
    elif key == "Nvidia":
        embeddings_model = NVIDIAEmbeddings(
        model=settings.EMBEDDINGS_MODEL[key][0],
        api_key=os.getenv("NVIDIA_API_KEY"),
    )
        
    elif key == "Cohere":
        embeddings_model = CohereEmbeddings(
        model=settings.EMBEDDINGS_MODEL[key][0],
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
        
    elif key == "Google":
        embeddings_model = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDINGS_MODEL[key][0],
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
        
    elif key == "Ollama":
        embeddings_model = OllamaEmbeddings(
        model=settings.EMBEDDINGS_MODEL[key][0],
        base_url=settings.OLLAMA_URL
    )
        
    vector_dim = settings.EMBEDDINGS_MODEL[key][1]   
    
    return (embeddings_model, vector_dim)



