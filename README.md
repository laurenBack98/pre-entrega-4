# RAG híbrido de políticas de Rapanui

Aplicación de generación aumentada por recuperación (RAG) que indexa información sobre Rapanui en Pinecone. El flujo principal recupera contexto mediante embeddings generados localmente con Ollama; además, incluye un recuperador híbrido que combina búsqueda vectorial y BM25 para la evaluación. Las respuestas del flujo principal se generan con Ollama y se validan con un esquema Pydantic.

## Requisitos

- Python 3.10 o superior
- [Ollama](https://ollama.com/) instalado y en ejecución
- Una cuenta y una API key de [Pinecone](https://www.pinecone.io/)

## Instalación

Desde la carpeta del proyecto, crea y activa un entorno virtual en PowerShell e instala las dependencias:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Descarga en Ollama el modelo de chat y el modelo de embeddings usados por defecto:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

## Configuración

Crea un archivo `.env` en la carpeta del proyecto con las credenciales de Pinecone y, opcionalmente, los ajustes de Ollama:

```dotenv
PINECONE_API_KEY=tu-api-key
PINECONE_INDEX_NAME=rag-pinecone-768
LLM_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://localhost:11434
```

No compartas ni subas el archivo `.env` al control de versiones.

| Variable | Obligatoria | Valor por defecto | Descripción |
|---|---:|---|---|
| `PINECONE_API_KEY` | Sí | — | API key para acceder a Pinecone |
| `PINECONE_INDEX_NAME` | Sí | — | Nombre del índice vectorial |
| `LLM_MODEL` | No | `qwen2.5:3b` | Modelo de chat disponible en Ollama |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Dirección del servidor Ollama |

El índice se crea automáticamente si no existe, con dimensión 768, métrica coseno y especificación serverless de AWS en `us-east-1`. Los embeddings se generan con `nomic-embed-text`.

## Ejecución

Con el entorno virtual activado, Ollama en ejecución y las variables configuradas:

```powershell
python main.py
```

El programa construye el vectorstore en Pinecone, ejecuta una consulta con información disponible y otra sin información suficiente, y muestra la respuesta estructurada junto con sus fuentes.

## Evaluación del recuperador

El archivo `golden_dataset.json` contiene consultas de prueba, fuentes esperadas y palabras clave. Para evaluar el recuperador híbrido y ver las métricas medias de Precision@k y Recall@k:

```powershell
python evaluate.py
```

La evaluación utiliza Pinecone para la recuperación semántica y BM25 en memoria para la recuperación léxica.

### Resultados de evaluación

Se evaluó el rendimiento del `EnsembleRetriever` (Pinecone + BM25, con pesos 50/50) sobre un Golden Dataset de cuatro consultas representativas, con `k=3`:

- **Media Recall@3:** 100.00% en el conjunto evaluado.
- **Media Precision@3:** 58.33% en el conjunto evaluado.

### Control de alucinaciones

El pipeline también se probó con preguntas fuera del dominio de los documentos. Cuando la información no aparece en el contexto recuperado, responde «No lo sé».

## Datos e implementación

Los documentos de ejemplo y la lógica de creación del índice están en `ingesta.py`. Los textos se dividen en fragmentos de 500 caracteres con un solapamiento de 50 y se cargan en Pinecone.

- `main.py`: ejecuta el flujo RAG con recuperación vectorial.
- `chain.py`: construye el prompt y la cadena RAG con salida Pydantic.
- `retriver.py`: combina la búsqueda vectorial de Pinecone y BM25 mediante `EnsembleRetriever`.
- `evaluate.py`: calcula Precision@k y Recall@k usando el conjunto de prueba.
- `scherma.py`: define la configuración del modelo y el formato de respuesta.
