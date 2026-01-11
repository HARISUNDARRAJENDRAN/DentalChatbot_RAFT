"""
PDF Extractor for Dental Textbooks
Extracts text and metadata from PDF files using PyMuPDF.
"""

import fitz  # PyMuPDF
import json
from pathlib import Path
from typing import Generator
from dataclasses import dataclass, asdict
from tqdm import tqdm


@dataclass
class ExtractedPage:
    """Represents a single extracted page from a PDF."""
    text: str
    source: str
    page_number: int
    total_pages: int
    chapter: str | None = None


@dataclass
class ExtractedDocument:
    """Represents an entire extracted PDF document."""
    source: str
    total_pages: int
    pages: list[ExtractedPage]


class PDFExtractor:
    """Extract text from dental PDF textbooks."""

    def __init__(self, min_text_length: int = 100):
        """
        Initialize the PDF extractor.

        Args:
            min_text_length: Minimum characters per page to consider valid
        """
        self.min_text_length = min_text_length

    def extract_page(self, page: fitz.Page, source: str,
                     page_num: int, total_pages: int) -> ExtractedPage:
        """
        Extract text from a single PDF page.

        Args:
            page: PyMuPDF page object
            source: Source filename
            page_num: Current page number (1-indexed)
            total_pages: Total pages in document

        Returns:
            ExtractedPage with text and metadata
        """
        # Extract text with layout preservation
        text = page.get_text("text")

        # Clean up whitespace
        text = " ".join(text.split())

        return ExtractedPage(
            text=text,
            source=source,
            page_number=page_num,
            total_pages=total_pages
        )

    def extract_document(self, pdf_path: Path | str) -> ExtractedDocument:
        """
        Extract all pages from a PDF document.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ExtractedDocument with all pages
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)

        pages = []
        for page_num, page in enumerate(doc, start=1):
            extracted = self.extract_page(
                page=page,
                source=pdf_path.name,
                page_num=page_num,
                total_pages=len(doc)
            )

            # Only include pages with sufficient text
            if len(extracted.text) >= self.min_text_length:
                pages.append(extracted)

        doc.close()

        return ExtractedDocument(
            source=pdf_path.name,
            total_pages=len(doc),
            pages=pages
        )

    def extract_directory(self,
                          input_dir: Path | str,
                          output_path: Path | str | None = None,
                          show_progress: bool = True) -> list[ExtractedDocument]:
        """
        Extract all PDFs from a directory.

        Args:
            input_dir: Directory containing PDF files
            output_path: Optional path to save extracted text as JSONL
            show_progress: Show progress bar

        Returns:
            List of ExtractedDocument objects
        """
        input_dir = Path(input_dir)
        pdf_files = list(input_dir.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(f"No PDF files found in {input_dir}")

        documents = []
        iterator = tqdm(pdf_files, desc="Extracting PDFs") if show_progress else pdf_files

        for pdf_path in iterator:
            try:
                doc = self.extract_document(pdf_path)
                documents.append(doc)
            except Exception as e:
                print(f"Error extracting {pdf_path.name}: {e}")
                continue

        # Save to JSONL if output path provided
        if output_path:
            self.save_to_jsonl(documents, output_path)

        return documents

    def save_to_jsonl(self, documents: list[ExtractedDocument],
                      output_path: Path | str) -> None:
        """
        Save extracted documents to JSONL format.

        Args:
            documents: List of extracted documents
            output_path: Path to output JSONL file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                for page in doc.pages:
                    record = asdict(page)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Saved {sum(len(d.pages) for d in documents)} pages to {output_path}")

    def iter_pages(self,
                   input_dir: Path | str) -> Generator[ExtractedPage, None, None]:
        """
        Iterate over all pages from all PDFs in a directory.

        Args:
            input_dir: Directory containing PDF files

        Yields:
            ExtractedPage objects one at a time
        """
        input_dir = Path(input_dir)

        for pdf_path in input_dir.glob("*.pdf"):
            try:
                doc = self.extract_document(pdf_path)
                for page in doc.pages:
                    yield page
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {e}")
                continue


class GoogleDrivePDFExtractor(PDFExtractor):
    """
    PDF Extractor with Google Drive support.

    For use in Google Colab with mounted Drive.
    """

    # Default Google Drive paths
    DRIVE_MOUNT = "/content/drive"
    DRIVE_BOOKS_FOLDER = "/content/drive/MyDrive/MyDentalBooks"
    DRIVE_OUTPUT_FOLDER = "/content/drive/MyDrive/RAFT_dental_data"

    @staticmethod
    def mount_drive() -> bool:
        """Mount Google Drive in Colab."""
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            print("Google Drive mounted successfully!")
            return True
        except ImportError:
            print("Not running in Google Colab. Using local paths.")
            return False
        except Exception as e:
            print(f"Failed to mount Drive: {e}")
            return False

    @classmethod
    def from_drive(
        cls,
        books_folder: str | None = None,
        output_folder: str | None = None,
        mount: bool = True
    ) -> "GoogleDrivePDFExtractor":
        """
        Create extractor configured for Google Drive.

        Args:
            books_folder: Path to dental books in Drive
            output_folder: Path for output files in Drive
            mount: Whether to mount Drive first

        Returns:
            Configured GoogleDrivePDFExtractor
        """
        if mount:
            cls.mount_drive()

        extractor = cls()
        extractor.books_folder = Path(books_folder or cls.DRIVE_BOOKS_FOLDER)
        extractor.output_folder = Path(output_folder or cls.DRIVE_OUTPUT_FOLDER)

        # Ensure output folder exists
        extractor.output_folder.mkdir(parents=True, exist_ok=True)

        return extractor

    def extract_from_drive(
        self,
        books_folder: str | None = None,
        output_path: str | None = None,
        show_progress: bool = True
    ) -> list[ExtractedDocument]:
        """
        Extract all PDFs from Google Drive folder.

        Args:
            books_folder: Override books folder path
            output_path: Override output path
            show_progress: Show progress bar

        Returns:
            List of ExtractedDocument objects
        """
        input_dir = Path(books_folder) if books_folder else self.books_folder

        if output_path:
            out_path = Path(output_path)
        else:
            out_path = self.output_folder / "raw_pages.jsonl"

        print(f"Extracting PDFs from: {input_dir}")
        print(f"Output: {out_path}")

        return self.extract_directory(
            input_dir=input_dir,
            output_path=out_path,
            show_progress=show_progress
        )


def main():
    """Example usage of PDF extractor."""
    import sys

    # Check if running in Colab
    IN_COLAB = 'google.colab' in sys.modules

    if IN_COLAB:
        # Use Google Drive
        print("Running in Google Colab - using Google Drive")
        extractor = GoogleDrivePDFExtractor.from_drive(
            books_folder="/content/drive/MyDrive/MyDentalBooks",
            output_folder="/content/drive/MyDrive/RAFT_dental_data"
        )
        documents = extractor.extract_from_drive()
    else:
        # Use local paths
        print("Running locally")
        extractor = PDFExtractor(min_text_length=100)

        input_dir = Path("data/raw/dental_books")
        output_path = Path("data/processed/chunks/raw_pages.jsonl")

        if input_dir.exists():
            documents = extractor.extract_directory(
                input_dir=input_dir,
                output_path=output_path,
                show_progress=True
            )
        else:
            print(f"Directory {input_dir} not found.")
            print("For Google Colab, run this script in Colab with Drive mounted.")
            return

    print(f"Extracted {len(documents)} documents")
    total_pages = sum(len(d.pages) for d in documents)
    print(f"Total pages with text: {total_pages}")


if __name__ == "__main__":
    main()
