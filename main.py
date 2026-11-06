import asyncio
import os
from dotenv import load_dotenv
from chain import build_rag_chain
from factory import create_llm
from scherma import LLMConfig
from ingesta import build_vectorstore


async def main():
    load_dotenv()
    # 1. Configuración del modelo local
    modelo = os.getenv("LLM_MODEL", "qwen2.5:3b")
    print(f"Iniciando pipeline con modelo: {modelo}")
    config = LLMConfig(
        model=modelo,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.0,
        max_tokens=600,
    )
    llm = create_llm(config)
    # 2. Inicialización de la base vectorial persistente
    vectorstore, _ = build_vectorstore()
    # 3. Construcción de la cadena RAG con LCEL y Pydantic
    rag_chain = build_rag_chain(vectorstore, llm)
    # 4. Caso 1: Pregunta dentro del contexto
    pregunta_valida = "¿Cuáles son las políticas de calidad de rapanui?"
    print(f"\n--- Consulta 1 (Información presente) ---")
    print(f"Pregunta: {pregunta_valida}\n")
    
    res1 = await rag_chain.ainvoke(pregunta_valida)
    print("Respuesta:", res1.respuesta)
    print("Fuentes usadas:", res1.fuentes_usadas)
    # 5. Caso 2: Pregunta fuera del contexto (prueba de abstención)
    pregunta_invalida = "¿Cuál es la cotización de las acciones de la empresa en Wall Street?"
    print(f"\n--- Consulta 2 (Información ausente) ---")
    print(f"Pregunta: {pregunta_invalida}\n")
    
    res2 = await rag_chain.ainvoke(pregunta_invalida)
    print("Respuesta:", res2.respuesta)
    print("Fuentes usadas:", res2.fuentes_usadas)
if __name__ == "__main__":
    asyncio.run(main())