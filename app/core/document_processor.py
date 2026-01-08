import hashlib
import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Tuple

from pypdf import PdfReader

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class Chunk:
    """Represents a text chunk from a document."""

    chunk_id: str
    document_id: str
    text: str
    page_number: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    # Position tracking for highlighting
    char_start: int = 0  # Start position in page text
    char_end: int = 0  # End position in page text


class DocumentProcessor:
    """Handles PDF extraction and text chunking."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> List[Tuple[int, str]]:
        """
        Extract text from PDF bytes.

        Returns:
            List of (page_number, text) tuples (1-indexed)
        """
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []

        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = self._clean_text(text)
            if text.strip():
                pages.append((i, text))

        logger.info(f"Extracted {len(pages)} pages with text")
        return pages

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        text = " ".join(text.split())
        text = text.replace("\x00", "")
        return text.strip()

    def chunk_text(
        self,
        text: str,
        page_number: int,
        document_id: str,
        start_chunk_index: int = 0,
    ) -> List[Chunk]:
        """
        Split text into overlapping chunks with position tracking.

        Args:
            text: Text to chunk
            page_number: Source page number
            document_id: Parent document ID
            start_chunk_index: Starting index for chunk numbering

        Returns:
            List of Chunk objects with position metadata
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        chunk_index = start_chunk_index

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                for sep in [". ", "! ", "? ", "\n"]:
                    last_sep = chunk_text.rfind(sep)
                    if last_sep > self.chunk_size * 0.5:
                        chunk_text = chunk_text[: last_sep + 1]
                        end = start + len(chunk_text)
                        break

            # Track actual positions before stripping
            actual_start = start

            chunk_text_stripped = chunk_text.strip()

            # Adjust positions for leading whitespace
            leading_ws = len(chunk_text) - len(chunk_text.lstrip())
            actual_start += leading_ws
            actual_end = actual_start + len(chunk_text_stripped)

            if chunk_text_stripped:
                chunk_id = self._generate_chunk_id(document_id, chunk_index)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=chunk_text_stripped,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        char_start=actual_start,
                        char_end=actual_end,
                        metadata={
                            "page": page_number,
                            "chunk_index": chunk_index,
                            "char_count": len(chunk_text_stripped),
                            "char_start": actual_start,
                            "char_end": actual_end,
                        },
                    )
                )
                chunk_index += 1

            # Move forward with overlap
            start = end - self.chunk_overlap
            if start >= len(text) or end >= len(text):
                break

        return chunks

    def process_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
    ) -> Tuple[str, List[Chunk], int, Dict[int, str]]:
        """
        Process a PDF file: extract text and create chunks.

        Args:
            pdf_bytes: Raw PDF bytes
            filename: Original filename

        Returns:
            Tuple of (document_id, chunks, page_count, page_texts)
            page_texts: Dict mapping page_number -> full page text (for highlighting)
        """
        document_id = self._generate_document_id(pdf_bytes, filename)

        # Extract text from PDF
        pages = self.extract_text_from_pdf(pdf_bytes)
        page_count = len(pages)

        # Store page texts for highlighting reference
        page_texts: Dict[int, str] = dict(pages)

        # Chunk all pages
        all_chunks = []
        chunk_index = 0

        for page_num, page_text in pages:
            page_chunks = self.chunk_text(
                text=page_text,
                page_number=page_num,
                document_id=document_id,
                start_chunk_index=chunk_index,
            )
            all_chunks.extend(page_chunks)
            chunk_index += len(page_chunks)

        for chunk in all_chunks:
            chunk.metadata["filename"] = filename

        logger.info(f"Processed '{filename}': {page_count} pages, {len(all_chunks)} chunks")

        return document_id, all_chunks, page_count, page_texts

    def _generate_document_id(self, content: bytes, filename: str) -> str:
        """Generate unique document ID from content hash."""
        hash_input = filename.encode() + content[:1024]
        return hashlib.sha256(hash_input).hexdigest()[:16]

    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        return f"{document_id}_{chunk_index:04d}"
