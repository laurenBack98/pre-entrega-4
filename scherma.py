from pydantic import BaseModel, Field, SecretStr
from typing import Optional

class LLMConfig(BaseModel):
    model: str
    base_url: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=500, ge=1)
    api_key: Optional[SecretStr] = None

class RespuestaRAG(BaseModel):
    respuesta: str = Field(
        description="Respuesta redactada basada exclusivamente en el contexto recuperado."
    )
    fuentes_usadas: list[str] = Field(
        default_factory=list,
        description="Lista de nombres de archivos o títulos de los fragmentos utilizados."
    )
    