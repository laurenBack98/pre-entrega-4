import json
import os
from ingesta import build_vectorstore
import retriver


def load_golden_dataset(filepath: str = "golden_dataset.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def is_relevant(doc, item) -> bool:
    """
    Evalúa si un documento recuperado es relevante comparando
    la fuente de origen o las palabras clave esperadas.
    """
    doc_source = doc.metadata.get("source", "")
    content = doc.page_content.lower()

    # 1. Match por metadato de fuente
    if doc_source in item.get("expected_sources", []):
        return True

    # 2. Match por palabras clave
    for kw in item.get("expected_keywords", []):
        if kw.lower() in content:
            return True

    return False


def calculate_metrics(golden_set, retriever, k: int = 3):
    total_precision = 0.0
    total_recall = 0.0

    print("=" * 60)
    print(f" EVALUACIÓN DEL RECUPERADOR HÍBRIDO (k={k})")
    print("=" * 60)

    for idx, item in enumerate(golden_set, 1):
        query = item["query"]
        expected_sources = item.get("expected_sources", [])

        # Recuperar top-k documentos
        retrieved_docs = retriever.invoke(query)[:k]

        # Contar relevantes recuperados
        relevant_retrieved = sum(1 for doc in retrieved_docs if is_relevant(doc, item))

        # Precision@k: relevantes recuperados / total recuperados (k)
        precision_at_k = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0.0

        # Recall@k: relevantes recuperados / total relevantes esperados (mínimo 1)
        total_relevant_expected = max(len(expected_sources), 1)
        recall_at_k = min(relevant_retrieved / total_relevant_expected, 1.0)

        total_precision += precision_at_k
        total_recall += recall_at_k

        print(f"\n[{idx}] Consulta: {query}")
        print(f"    Precision@{k}: {precision_at_k:.2f} | Recall@{k}: {recall_at_k:.2f}")
        for r_idx, doc in enumerate(retrieved_docs, 1):
            rel_flag = "✓ Relevante" if is_relevant(doc, item) else "✗ No relevante"
            print(f"      - Doc {r_idx} [{doc.metadata.get('source')}]: {rel_flag}")

    n = len(golden_set)
    mean_precision = total_precision / n if n > 0 else 0.0
    mean_recall = total_recall / n if n > 0 else 0.0

    print("\n" + "=" * 60)
    print(" RESULTADOS GLOBALES")
    print(f" Media Precision@{k}: {mean_precision:.2%}")
    print(f" Media Recall@{k}:    {mean_recall:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    print("Cargando chunks y configurando EnsembleRetriever...")
    _, chunks = build_vectorstore()
    retriever = retriver.get_hybrid_retriever(chunks, k=3)

    dataset_path = "golden_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Error: no se encontró el archivo {dataset_path}")
    else:
        golden_set = load_golden_dataset(dataset_path)
        calculate_metrics(golden_set, retriever, k=3)
