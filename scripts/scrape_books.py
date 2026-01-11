"""
Dental Books Scraper
Scrapes dental books from http://43.230.198.52/lib/book/ and saves to Google Drive.

Usage:
    1. Run this script in Google Colab with Drive mounted
    2. Or run locally and upload to Google Drive afterwards
"""

import os
import re
import time
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
from typing import Generator
from dataclasses import dataclass
from tqdm import tqdm
from bs4 import BeautifulSoup
import concurrent.futures


@dataclass
class BookInfo:
    """Information about a book to download."""
    title: str
    url: str
    category: str = ""
    file_type: str = "pdf"


class DentalBooksScraper:
    """
    Scraper for dental books library.

    Designed to scrape from http://43.230.198.52/lib/book/
    """

    BASE_URL = "http://43.230.198.52/lib/book/"

    def __init__(
        self,
        output_dir: str = "dental_books",
        delay: float = 1.0,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the scraper.

        Args:
            output_dir: Directory to save downloaded books
            delay: Delay between requests (be respectful to server)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed downloads
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries

        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })

    def get_page(self, url: str) -> BeautifulSoup | None:
        """
        Fetch and parse a webpage.

        Args:
            url: URL to fetch

        Returns:
            BeautifulSoup object or None on failure
        """
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return BeautifulSoup(response.content, "html.parser")
            except requests.RequestException as e:
                print(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))
        return None

    def discover_books(self) -> list[BookInfo]:
        """
        Discover all available books from the library.

        Returns:
            List of BookInfo objects
        """
        books = []

        print(f"Fetching book list from {self.BASE_URL}")
        soup = self.get_page(self.BASE_URL)

        if soup is None:
            print("Failed to fetch main page. Trying alternative discovery...")
            return self._discover_by_crawling()

        # Strategy 1: Look for direct PDF links
        for link in soup.find_all("a", href=True):
            href = link["href"]

            # Check if it's a PDF link
            if href.lower().endswith(".pdf"):
                full_url = urljoin(self.BASE_URL, href)
                title = self._extract_title(link, href)
                books.append(BookInfo(title=title, url=full_url))

        # Strategy 2: Look for book listing pages
        for link in soup.find_all("a", href=True):
            href = link["href"]

            # Check if it's a category or book page
            if "/book/" in href and not href.endswith(".pdf"):
                page_url = urljoin(self.BASE_URL, href)

                # Fetch sub-page for PDFs
                sub_soup = self.get_page(page_url)
                if sub_soup:
                    for pdf_link in sub_soup.find_all("a", href=True):
                        pdf_href = pdf_link["href"]
                        if pdf_href.lower().endswith(".pdf"):
                            full_url = urljoin(page_url, pdf_href)
                            title = self._extract_title(pdf_link, pdf_href)

                            # Avoid duplicates
                            if not any(b.url == full_url for b in books):
                                books.append(BookInfo(title=title, url=full_url))

                time.sleep(self.delay)

        # Strategy 3: Look for common patterns
        # Many library sites use numbered pages or alphabetical listing
        for pattern in self._generate_common_patterns():
            page_url = urljoin(self.BASE_URL, pattern)
            sub_soup = self.get_page(page_url)

            if sub_soup:
                for pdf_link in sub_soup.find_all("a", href=True):
                    pdf_href = pdf_link["href"]
                    if pdf_href.lower().endswith(".pdf"):
                        full_url = urljoin(page_url, pdf_href)
                        title = self._extract_title(pdf_link, pdf_href)

                        if not any(b.url == full_url for b in books):
                            books.append(BookInfo(title=title, url=full_url))

            time.sleep(self.delay)

        print(f"Discovered {len(books)} books")
        return books

    def _discover_by_crawling(self) -> list[BookInfo]:
        """Fallback: crawl the site to find books."""
        books = []
        visited = set()
        to_visit = [self.BASE_URL]

        while to_visit and len(visited) < 100:  # Limit crawl depth
            url = to_visit.pop(0)

            if url in visited:
                continue
            visited.add(url)

            soup = self.get_page(url)
            if soup is None:
                continue

            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(url, href)

                # Check if PDF
                if href.lower().endswith(".pdf"):
                    title = self._extract_title(link, href)
                    if not any(b.url == full_url for b in books):
                        books.append(BookInfo(title=title, url=full_url))

                # Add to crawl queue if same domain
                elif self._is_same_domain(full_url) and full_url not in visited:
                    to_visit.append(full_url)

            time.sleep(self.delay)

        return books

    def _is_same_domain(self, url: str) -> bool:
        """Check if URL is on the same domain."""
        base_domain = urlparse(self.BASE_URL).netloc
        url_domain = urlparse(url).netloc
        return base_domain == url_domain

    def _generate_common_patterns(self) -> list[str]:
        """Generate common URL patterns for library sites."""
        patterns = []

        # Pagination
        for i in range(1, 20):
            patterns.append(f"?page={i}")
            patterns.append(f"page/{i}/")
            patterns.append(f"index_{i}.html")

        # Categories (common dental categories)
        categories = [
            "endodontics", "orthodontics", "periodontics", "prosthodontics",
            "oral-surgery", "pediatric-dentistry", "dental-anatomy",
            "oral-pathology", "dental-materials", "radiology",
            "pharmacology", "oral-medicine", "implants"
        ]
        for cat in categories:
            patterns.append(f"{cat}/")
            patterns.append(f"category/{cat}/")

        return patterns

    def _extract_title(self, link_element, href: str) -> str:
        """Extract book title from link or filename."""
        # Try link text first
        text = link_element.get_text(strip=True)
        if text and len(text) > 5 and not text.endswith(".pdf"):
            return self._clean_title(text)

        # Fall back to filename
        filename = unquote(Path(urlparse(href).path).stem)
        return self._clean_title(filename)

    def _clean_title(self, title: str) -> str:
        """Clean up book title."""
        # Remove common prefixes/suffixes
        title = re.sub(r"^\d+[\._-]?\s*", "", title)  # Leading numbers
        title = re.sub(r"[\._-]+", " ", title)  # Replace separators with spaces
        title = re.sub(r"\s+", " ", title)  # Normalize whitespace
        return title.strip()[:200]  # Limit length

    def download_book(self, book: BookInfo) -> bool:
        """
        Download a single book.

        Args:
            book: BookInfo object

        Returns:
            True if successful, False otherwise
        """
        # Create safe filename
        safe_title = re.sub(r'[<>:"/\\|?*]', "_", book.title)
        filename = f"{safe_title}.pdf"
        filepath = self.output_dir / filename

        # Skip if already exists
        if filepath.exists():
            print(f"Already exists: {filename}")
            return True

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    book.url,
                    timeout=self.timeout * 2,  # Longer timeout for downloads
                    stream=True
                )
                response.raise_for_status()

                # Check if it's actually a PDF
                content_type = response.headers.get("Content-Type", "")
                if "pdf" not in content_type.lower() and not book.url.lower().endswith(".pdf"):
                    print(f"Not a PDF: {book.title}")
                    return False

                # Download with progress
                total_size = int(response.headers.get("Content-Length", 0))

                with open(filepath, "wb") as f:
                    if total_size > 0:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    else:
                        f.write(response.content)

                print(f"Downloaded: {filename}")
                return True

            except requests.RequestException as e:
                print(f"Attempt {attempt + 1} failed for {book.title}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))

        return False

    def download_all(
        self,
        books: list[BookInfo] | None = None,
        max_workers: int = 3
    ) -> tuple[int, int]:
        """
        Download all discovered books.

        Args:
            books: List of books to download (discovers if None)
            max_workers: Number of concurrent downloads

        Returns:
            Tuple of (successful, failed) counts
        """
        if books is None:
            books = self.discover_books()

        if not books:
            print("No books found to download")
            return 0, 0

        print(f"\nDownloading {len(books)} books to {self.output_dir}")

        successful = 0
        failed = 0

        # Sequential download (safer, more respectful)
        for book in tqdm(books, desc="Downloading"):
            if self.download_book(book):
                successful += 1
            else:
                failed += 1
            time.sleep(self.delay)

        print(f"\nComplete! Downloaded: {successful}, Failed: {failed}")
        return successful, failed

    def get_existing_books(self) -> set[str]:
        """Get set of already downloaded book filenames."""
        return {f.stem for f in self.output_dir.glob("*.pdf")}


class GoogleDriveScraper(DentalBooksScraper):
    """
    Scraper with Google Drive integration.

    For use in Google Colab with mounted Drive.
    """

    def __init__(
        self,
        drive_folder: str = "/content/drive/MyDrive/MyDentalBooks",
        **kwargs
    ):
        """
        Initialize with Google Drive path.

        Args:
            drive_folder: Path to Google Drive folder
        """
        super().__init__(output_dir=drive_folder, **kwargs)

    @staticmethod
    def mount_drive():
        """Mount Google Drive in Colab."""
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            print("Google Drive mounted successfully!")
            return True
        except ImportError:
            print("Not running in Google Colab. Drive mounting skipped.")
            return False
        except Exception as e:
            print(f"Failed to mount Drive: {e}")
            return False

    def sync_to_drive(self, local_dir: str = None):
        """
        Sync local downloads to Google Drive.

        For use when running locally and want to upload to Drive.
        """
        if local_dir is None:
            print("Specify local directory to sync from")
            return

        local_path = Path(local_dir)

        for pdf in local_path.glob("*.pdf"):
            target = self.output_dir / pdf.name
            if not target.exists():
                import shutil
                shutil.copy2(pdf, target)
                print(f"Copied: {pdf.name}")


def scrape_dental_books(
    output_dir: str = "dental_books",
    use_google_drive: bool = False,
    drive_folder: str = "/content/drive/MyDrive/MyDentalBooks"
) -> tuple[int, int]:
    """
    Main function to scrape dental books.

    Args:
        output_dir: Local output directory
        use_google_drive: Use Google Drive for storage
        drive_folder: Google Drive folder path

    Returns:
        Tuple of (successful, failed) download counts
    """
    if use_google_drive:
        scraper = GoogleDriveScraper(drive_folder=drive_folder)
        GoogleDriveScraper.mount_drive()
    else:
        scraper = DentalBooksScraper(output_dir=output_dir)

    # Show existing books
    existing = scraper.get_existing_books()
    print(f"Found {len(existing)} existing books")

    # Discover and download
    books = scraper.discover_books()

    # Filter out already downloaded
    new_books = [b for b in books if scraper._clean_title(b.title) not in existing]
    print(f"New books to download: {len(new_books)}")

    if new_books:
        return scraper.download_all(new_books)
    else:
        print("All books already downloaded!")
        return 0, 0


if __name__ == "__main__":
    # Run scraper
    scrape_dental_books(
        output_dir="dental_books",
        use_google_drive=False  # Set True in Colab
    )
