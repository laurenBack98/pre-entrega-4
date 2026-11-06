import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-pinecone-768")


def get_hybrid_retriever(chunks, k: int = 5, weights=(0.5, 0.5)):
    """
    Construye un EnsembleRetriever combinando Pinecone (Semántico)
    y BM25 (Léxico) sobre los fragmentos procesados.
    """
    # 1. Embeddings locales
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 2. Retriever vectorial desde el índice existente en Pinecone
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 3. Retriever léxico (BM25) en memoria
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # 4. Ensamble híbrido con pesos balanceados
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=list(weights)
    )

    return ensemble_retriever


if __name__ == "__main__":
    # Test rápido de integración
    from ingestion import build_vectorstore

    print("Verificando índice y cargando chunks...")
    _, chunks = build_vectorstore()

    print("Inicializando EnsembleRetriever...")
    retriever = get_hybrid_retriever(chunks, k=3)

    query = "¿Qué norma ISO certifica la política de calidad de Rapanui?"
    docs = retriever.invoke(query)

    print(f"\nResultados recuperados para: '{query}'\n")
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] Fuente: {doc.metadata.get('source')}")
        print(f"    Texto: {doc.page_content[:120]}...\n")
