"""In-memory PDF extraction, chunking, and FAISS indexing."""

from io import BytesIO
from typing import Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

CHUNK_SIZE = 900
CHUNK_OVERLAP = 140

MAX_DOCUMENT_COUNT = 8
MAX_UPLOAD_BYTES = 30 * 1024 * 1024

def extract_pdf_pages(file_bytes: bytes, filename: str, document_type: str) -> list[Document]:
    if not file_bytes:
        raise ValueError(f"{filename} is empty.")
    try:
        reader = PdfReader(BytesIO(file_bytes))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError(f"{filename} is password protected.")
        pages = [
            Document(
                page_content=(page.extract_text() or "").strip(),
                metadata={"source": filename, "page": index + 1, "document_type": document_type},
            )
            for index, page in enumerate(reader.pages)
        ]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read {filename} as a valid PDF.") from exc
    pages = [page for page in pages if page.page_content]
    if not pages:
        raise ValueError(f"No selectable text was found in {filename}. Scanned PDFs need OCR first.")
    return pages


def extract_uploaded_files(resume_file: Any | None, study_files: list[Any]) -> list[Document]:
    files = ([resume_file] if resume_file is not None else []) + study_files
    if not files:
        raise ValueError("Select at least one PDF to index.")
    if len(files) > MAX_DOCUMENT_COUNT:
        raise ValueError(f"Select no more than {MAX_DOCUMENT_COUNT} PDFs per indexing run.")
    if sum(int(file.size) for file in files) > MAX_UPLOAD_BYTES:
        raise ValueError("Selected PDFs exceed the 30 MB total upload limit.")

    pages: list[Document] = []
    if resume_file is not None:
        pages.extend(extract_pdf_pages(resume_file.getvalue(), resume_file.name, "resume"))
    for file in study_files:
        pages.extend(extract_pdf_pages(file.getvalue(), file.name, "study"))
    return pages


def split_documents(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(pages)


def create_vector_store(
    pages: list[Document], embeddings: HuggingFaceEmbeddings
) -> tuple[FAISS, int]:
    chunks = split_documents(pages)
    if not chunks:
        raise ValueError("No document text was available to index.")
    return FAISS.from_documents(chunks, embeddings), len(chunks)


def retrieve(vector_store: FAISS, query: str, *, count: int = 5) -> list[Document]:
    return vector_store.similarity_search(query, k=count)


def format_context(documents: list[Document], *, max_chars: int = 10000) -> str:
    sections: list[str] = []
    remaining = max_chars
    for document in documents:
        metadata: dict[str, Any] = document.metadata
        label = f"{metadata.get('source', 'Uploaded document')}, page {metadata.get('page', '?')}"
        section = f"[{label}]\n{document.page_content[:remaining]}"
        sections.append(section)
        remaining -= len(section)
        if remaining <= 0:
            break
    return "\n\n".join(sections)