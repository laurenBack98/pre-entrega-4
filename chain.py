from scherma import RespuestaRAG
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    """Une los chunks recuperados en un solo string de contexto."""
    return "\n\n".join(
        f"Fuente: {doc.metadata.get('source', 'desconocida')}\n{doc.page_content}"
        for doc in docs
    )

def build_rag_chain(vectorstore, llm):
    # 1. Definir la capa Retriever (k entre 3 y 5 según la consigna)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 2. Configurar el parser con tu schema de Pydantic
    parser = PydanticOutputParser(pydantic_object=RespuestaRAG)
    # 3. Prompt con instrucción de no inventar ("No lo sé")
    prompt = PromptTemplate(
        template=(
            "Eres un asistente técnico. Responde basándote ÚNICAMENTE en el siguiente contexto.\n"
            "Si la información no está en el contexto, di 'No lo sé'.\n\n"
            "Incluye en fuentes_usadas las fuentes que aparecen en el contexto.\n\n"
            "Contexto:\n{context}\n\n"
            "Pregunta: {question}\n\n"
            "{format_instructions}"
        ),
        input_variables=["context", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    # 4. Cadena LCEL conectando el retriever
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
    return rag_chain