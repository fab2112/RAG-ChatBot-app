# Settings

# Users authentication
"""USERS = [
    ("admin", "admin"),
    ("user", "1234@1234"),
]"""
USERS = None

# LLM response language
LANGUAGE = "Português (pt-Br)" # "English (en)"

# Vector Databases URLs
QDRANT_URL = "http://localhost:6333"
CHROMADB_URL = "http://localhost:8000"

# Ollama internal Docker url - binded to host by extra_hosts
OLLAMA_URL = "http://localhost:11434"

# Vector DataBases
RETRIEVER=VECSTORAGE = "qdrant"  # qdrant | pinecone | chromadb
DATABASE_NAME = "data-space-1"

# Pinecone settings
PINECONE_REGION = "us-east-1"
PINECONE_CLOUD = "aws"   

# Rate limit - embeddings process
EMBEDDINGS_RATELIMIT = 0.1

# Available LLM models
MODELS = (
    "Groq_meta-llama/llama-4-scout-17b-16e-instruct",
    "Groq_meta-llama/llama-4-maverick-17b-128e-instruct",
    "Groq_deepseek-r1-distill-llama-70b",
    "Groq_llama-3.3-70b-versatile",
    "Groq_llama-3.2-90b-text-preview",
    "Groq_llama-3.1-70b-versatile",
    "Groq_llama3-70b-8192",
    "Google_gemini-2.5-flash-preview-04-17",
    "Google_gemini-2.5-pro-exp-03-25",
    "Google_gemma-3-27b-it",
    "Google_gemini-2.0-flash-exp",
    "Google_gemini-exp-1121",
    "Google_gemini-exp-1114",
    "Google_gemini-1.5-pro-002",
    "Google_gemini-1.5-pro",
    "Google_gemini-1.5-flash-002",
    "Google_gemini-1.5-flash",
    "Google_gemini-2.0-flash-thinking-exp-01-21",
    "Ollama_gemma3:1b",
    "OpenAI_gpt-4o",
    "OpenAI_gpt-4o-mini",
    "Nvidia_qwen/qwq-32b",
)

# Available dense embeddings models 
EMBEDDINGS_MODEL = {
    #"Ollama":["nomic-embed-text", 768],
    #"Pinecone":["multilingual-e5-large", 1024],
    #"Nvidia":["NV-Embed-QA", 1024],
    #"Cohere":["embed-multilingual-v3.0", 1024],
    "Google":["models/embedding-001", 768]  
}
