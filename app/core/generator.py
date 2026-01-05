"""Generator service using Google Gemini API."""

import logging
from dataclasses import dataclass
from typing import List

from google import genai
from google.genai import types

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


class Generator:
    """Generate answers using Gemini with retrieved context."""

    def __init__(self):
        self._client = None
        self.model_name = "gemini-2.5-flash"

    @property
    def client(self) -> genai.Client:
        """Lazy load Gemini client."""
        if self._client is None:
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY not configured")
            self._client = genai.Client(api_key=settings.google_api_key)
            logger.info("Gemini client initialized")
        return self._client

    def generate(
        self,
        query: str,
        context: List[SearchResult],
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> GeneratedAnswer:
        """
        Generate answer using retrieved context.

        Args:
            query: User question
            context: Retrieved chunks for grounding
            max_tokens: Max response length
            temperature: Creativity (0-1)

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

        # Generate response
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        answer = response.text if response.text else "Unable to generate response."

        # Build sources list
        sources = [
            {
                "source_id": i,
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,  # noqa: PLR2004
                "relevance_score": chunk.score,
            }
            for i, chunk in enumerate(context, 1)
        ]

        logger.info(f"Generated answer for query: '{query[:50]}...'")
        return GeneratedAnswer(
            answer=answer,
            sources=sources,
            model=self.model_name,
        )


def get_generator() -> Generator:
    """Get generator instance."""
    return Generator()
