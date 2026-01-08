"""Generator service using LangChain for multi-provider LLM support."""

import logging
from dataclasses import dataclass
from typing import List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.core.retriever import SearchResult

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class GeneratedAnswer:
    """Generated answer with citations."""

    answer: str
    sources: List[dict]
    model: str


def get_llm(temperature: Optional[float] = None) -> BaseChatModel:
    """
    Get LLM based on configuration.

    Args:
        temperature: Optional temperature override

    Supports:
    - OpenAI (gpt-4, gpt-3.5-turbo, etc.)
    - Anthropic (claude-3, claude-2, etc.)
    - Google (gemini-pro, gemini-1.5-flash, etc.)
    - Groq (llama, mixtral, etc.)
    """
    provider = settings.llm_provider.lower()
    model = settings.llm_model
    temp = temperature if temperature is not None else settings.llm_temperature

    if provider == "openai":
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=temp,
            api_key=settings.openai_api_key,
        )

    elif provider == "anthropic":
        return ChatAnthropic(
            model=model or "claude-3-5-sonnet-20241022",
            temperature=temp,
            api_key=settings.anthropic_api_key,
        )

    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash-exp",
            temperature=temp,
            google_api_key=settings.google_api_key,
        )

    elif provider == "groq":
        return ChatGroq(
            model=model or "llama-3.3-70b-versatile",
            temperature=temp,
            api_key=settings.groq_api_key,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. Supported: openai, anthropic, google, groq, ollama, azure"
        )


class Generator:
    """Generate answers using LangChain with retrieved context."""

    def __init__(self) -> None:
        self._model_name: Optional[str] = None

    def _get_model_name(self) -> str:
        """Get model name for logging."""
        if self._model_name is None:
            self._model_name = f"{settings.llm_provider}/{settings.llm_model or 'default'}"
        return self._model_name

    def generate(
        self,
        query: str,
        context: List[SearchResult],
        temperature: Optional[float] = None,
    ) -> GeneratedAnswer:
        """
        Generate answer using retrieved context.

        Args:
            query: User question
            context: Retrieved chunks for grounding
            temperature: Optional temperature override

        Returns:
            GeneratedAnswer with citations
        """
        # Build context string with source markers
        context_parts = []
        for i, chunk in enumerate(context, 1):
            context_parts.append(
                f"[Source {i}] (Document: {chunk.document_id}, Page: {chunk.page_number})\n{chunk.text}"
            )
        context_str = "\n\n".join(context_parts)

        # System prompt for grounded generation
        system_prompt = """You are a helpful assistant that answers questions based on provided context.

Rules:
1. Only use information from the provided context
2. Cite sources using [Source N] format when using information
3. If the context doesn't contain enough information, say so
4. Be concise and accurate
5. Never make up information not in the context"""

        # User prompt
        user_prompt = f"""Context:
{context_str}

Question: {query}

Answer the question based on the context above. Cite sources using [Source N] format."""

        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Generate response with fresh LLM instance using specified temperature
        try:
            llm = get_llm(temperature=temperature)
            response = llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Error generating response: {str(e)}"

        # Build sources list with position info for highlighting
        sources = [
            {
                "source_id": i,
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "relevance_score": chunk.score,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }
            for i, chunk in enumerate(context, 1)
        ]

        logger.info(f"Generated answer for query: '{query[:50]}...'")
        return GeneratedAnswer(
            answer=answer,
            sources=sources,
            model=self._get_model_name(),
        )


def get_generator() -> Generator:
    """Get generator instance."""
    return Generator()
