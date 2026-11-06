import hashlib
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def build_vectorstore(documents=None):
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    dimension = 768  # nomic-embed-text

    # 1. Inicialización y creación de índice en Pinecone
    pc = Pinecone(api_key=api_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creando índice '{index_name}' en Pinecone...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            print("Esperando a que el índice esté disponible...")
            time.sleep(1)
        print(f"Índice '{index_name}' creado exitosamente.")

    # 2. Embeddings locales con Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 3. Documentos base si no se pasan por parámetro
    if documents is None:
        documentos = {
            "Politicas de calidad": """
• Estándares Internacionales: La empresa rige su producción bajo la Política de Calidad certificada por IRAM - ISO 9001-2015. 
Esta norma aplica desde el diseño y desarrollo hasta la comercialización de chocolates, pastelería y helados. 
[1] (https://www.rapanui.com.ar/politica).
• Seguridad Alimentaria y TACC: Cuenta con políticas estrictas de segregación en sus plantas 
(como la planta Vereertbrugghen) especializadas en la elaboración de alimentos libres de gluten 
(Sin TACC) para evitar la contaminación cruzada. Sus procesos también siguen lineamientos HACCP e ISO 22000. 
[1] (https://www.rapanui.com.ar/ingredientes-y-productores), [2] (https://www.rapanui.com.ar/politica).
• Abastecimiento Sustentable: Sus políticas de compras exigen que materias primas clave provengan de cooperativas familiares sin agroquímicos (cacao),
huevos de gallinas libres de jaula con certificación Certified Humane, 
y leche de pastoreo libre. 
[1] (https://www.rapanui.com.ar/ingredientes-y-productores)
""",
            "Recursos Humanos y Beneficios Laborales": """
Para sostener una plantilla que supera los 1.600 empleados a nivel global, 
sus políticas de contratación e incentivos en los locales comerciales 
(como los de CABA y Bariloche) estipulan: [1] (https://rapanui.hiringroom.com/jobs), 
[2] (https://www.youtube.com/watch?v=zWNytzaOXJ4&t=761).
• Modalidad de Trabajo: Para puestos operativos en locales (atención, mostrador, heladeros), 
aplican principalmente jornadas part-time con horarios rotativos, orientadas frecuentemente a estudiantes universitarios. 
[1] (https://www.bumeran.com.ar/empleos/dependiente-de-mostrador-rapanui-sucursales-caba-1118411153.html).
• Paquete de Beneficios Corporativos: De acuerdo con sus búsquedas oficiales en portales como Hiring Room y Bumeran, incluyen:
	• Cobertura médica a través de obra social corporativa.
	• 50% de descuento en la red de gimnasios Sport Club.
	• Descuentos exclusivos en productos de la marca.
	• Obsequios corporativos en fechas especiales. [1] (https://www.bumeran.com.ar/empleos/dependiente-de-mostrador-rapanui-sucursales-caba-1118411153.html)
""",
            "Comunicación Interna y Cultura Organizacional": """
• Digitalización de Procesos: Debido a la alta dispersión de su personal operativo en fábricas 
y locales que no utilizan computadoras en el día a día, Rapanui adoptó una política de democratización 
digital implementando la plataforma interna Humand. A través de esta aplicación móvil, centralizan la resolución de consultas de RR. HH., 
el envío de recibos de sueldo y las comunicaciones institucionales obligatorias. [1] (https://humand.co/es/caso-de-exito/rapanui-es/).
• Reinversión y Crecimiento: La dirección de la empresa sostiene la política interna 
de reinvertir la totalidad de las utilidades en la propia compañía (en lugar de diversificar en otros mercados financieros) 
para optimizar la eficiencia productiva en el contexto económico argentino. [1] (https://www.youtube.com/watch?v=R4dZSXbcb54)
""",
            "Política de Expansión Comercial": """
• No Franquiciar: Se mantiene una regla inquebrantable de exclusividad de propiedad. 
Cada punto de venta abierto en Argentina o el exterior es gestionado y financiado de manera directa por la casa central, 
asegurando homogeneidad en la atención, 
la estética de los locales y los precios. [1] (https://www.youtube.com/watch?v=zWNytzaOXJ4&t=761),
[2] (https://www.cronista.com/apertura/inversiones/la-estrategia-de-rapanui-que-va-al-reves-de-los-manuales-y-con-la-que-busca-facturar-600-millones/)
""",
        }
        raw_docs = [
            Document(page_content=content.strip(), metadata={"source": title})
            for title, content in documentos.items()
        ]
    else:
        raw_docs = documents

    # 4. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(raw_docs)

    # 5. Ingesta en Pinecone con IDs estables para evitar duplicados al reejecutar
    ids = [
        hashlib.sha256(
            f"{chunk.metadata.get('source', '')}\0{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        for chunk in chunks
    ]
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        ids=ids,
    )
    print(f"Se han agregado {len(chunks)} fragmentos al vectorstore de Pinecone.")
    return vectorstore, chunks


if __name__ == "__main__":
    build_vectorstore()
