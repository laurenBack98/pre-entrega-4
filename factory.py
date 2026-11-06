from langchain_ollama import ChatOllama
from scherma import LLMConfig


def create_llm(config: LLMConfig) -> ChatOllama:
    return ChatOllama(
        model=config.model,
        base_url=config.base_url or "http://localhost:11434",
        temperature=config.temperature,
        num_predict=config.max_tokens,
    )