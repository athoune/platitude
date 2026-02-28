"""
Scraper for Jacques Chirac speeches from vie-publique.fr
Fetches all speeches listed under Jacques Chirac and saves them to a JSON file.

Search is performed using the `field_intervenant_title` URL parameter, which
filters by speaker name – avoiding the noise of a full-text search.
"""

import time
import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.vie-publique.fr"

# Search URL filtered by speaker name (field_intervenant_title).
# Sorted by date (most recent first). Pagination is handled via &page=N.
SEARCH_URL = (
    "https://www.vie-publique.fr/discours/recherche"
    "?search_api_fulltext_discours="
    "&sort_by=field_date_prononciation_discour"
    "&field_intervenant_title=Jacques+Chirac"
    "&field_intervenant_qualite="
    "&field_date_prononciation_discour_interval%5Bmin%5D="
    "&field_date_prononciation_discour_interval%5Bmax%5D="
    "&page={page}"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ChiracSpeechScraper/1.0; "
        "+https://github.com/example/chirac-scraper)"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9",
}

REQUEST_DELAY = 0.5  # seconds between requests – be polite to the server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Speech:
    title: str
    url: str
    date: Optional[str]
    speech_type: Optional[str]
    full_text: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_soup(url: str, session: requests.Session) -> BeautifulSoup:
    """Fetch a URL and return a BeautifulSoup object."""
    response = session.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_listing_page(soup: BeautifulSoup) -> tuple[list[dict], bool]:
    """
    Parse a search results listing page and return:
    - list of dicts with 'title', 'url', 'date', 'speech_type'
    - bool indicating whether a next page exists

    The site renders results as <article> or <h3>/<h2> headings containing
    a link to /discours/<id>-<slug>, with date and type in sibling elements.
    """
    results = []

    # Primary strategy: find all links that point to a speech detail page.
    # Speech URLs match the pattern /discours/<numeric-id>-<slug>
    speech_links = soup.find_all("a", href=re.compile(r"^/discours/\d+-"))

    # Deduplicate while preserving order
    seen = set()
    for link in speech_links:
        href = link["href"]
        if href in seen:
            continue
        seen.add(href)

        title = link.get_text(strip=True)
        if not title:
            continue

        full_url = BASE_URL + href

        # Walk up the DOM to find the closest container holding metadata
        container = link.find_parent(["article", "li", "div"])
        date_text = None
        speech_type = None

        if container:
            # Date is typically in a <time> tag or a class containing "date"
            time_tag = container.find("time")
            if time_tag:
                date_text = time_tag.get("datetime") or time_tag.get_text(strip=True)
            else:
                date_node = container.find(
                    class_=re.compile(r"date|time|prononciation", re.I)
                )
                if date_node:
                    date_text = date_node.get_text(strip=True)

            # Speech type (Discours, Interview, Communiqué, etc.)
            type_node = container.find(
                class_=re.compile(r"type|nature|categor|label", re.I)
            )
            if type_node and type_node is not link:
                speech_type = type_node.get_text(strip=True)

        results.append(
            {
                "title": title,
                "url": full_url,
                "date": date_text,
                "speech_type": speech_type,
            }
        )

    # Detect a "next page" link (rel="next" or text matching "suivant")
    next_link = soup.find("a", rel="next") or soup.find(
        "a", string=re.compile(r"(suivant|next)", re.I)
    )
    # Also check for pagination links pointing to ?page=N
    if not next_link:
        pager = soup.find(class_=re.compile(r"pager|pagination", re.I))
        if pager:
            next_link = pager.find("a", string=re.compile(r"(suivant|next|›|»)", re.I))

    has_next = next_link is not None

    return results, has_next


def fetch_speech_text(url: str, session: requests.Session) -> Optional[str]:
    """
    Fetch the full text of a speech from its detail page.

    The speech body is contained in the Drupal field
    `.field--name-field-texte-integral`.
    """
    try:
        soup = get_soup(url, session)
        body = soup.select_one(".field--name-field-texte-integral")
        if body:
            return body.get_text(separator="\n", strip=True)

    except Exception as exc:
        logger.warning("Could not fetch speech text from %s: %s", url, exc)

    return None


# ---------------------------------------------------------------------------
# Main scraping logic
# ---------------------------------------------------------------------------


def collect_all_entries(session: requests.Session, max_pages: int = 100) -> list[dict]:
    """
    Paginate through search result pages and collect every speech entry
    (metadata only: title, url, date, speech_type).

    Args:
        max_pages: Maximum number of listing pages to fetch. Used as the
                   tqdm total so the bar shows determinate progress.

    A tqdm bar tracks progress with a known upper bound (max_pages).
    """
    all_entries: list[dict] = []
    seen_urls: set[str] = set()
    page = 0

    with tqdm(desc="Listing pages", unit="page", total=max_pages) as pbar:
        while page < max_pages:
            url = SEARCH_URL.format(page=page)
            try:
                soup = get_soup(url, session)
            except requests.RequestException as exc:
                logger.error("Failed to fetch listing page %d: %s", page, exc)
                break

            entries, has_next = parse_listing_page(soup)

            if not entries:
                logger.info("No entries on page %d – stopping pagination.", page)
                break

            for entry in entries:
                if entry["url"] not in seen_urls:
                    seen_urls.add(entry["url"])
                    all_entries.append(entry)

            pbar.update(1)
            pbar.set_postfix(speeches=len(all_entries))

            if not has_next:
                break

            page += 1
            time.sleep(REQUEST_DELAY)

    return all_entries


def scrape_all_speeches(
    fetch_full_text: bool = True,
    max_pages: int = 100,
    output_path: Path = Path("speeches.jsonl"),
    who: str = "Jacques Chirac",
) -> int:
    """
    Scrape all Jacques Chirac speeches from vie-publique.fr and write
    results incrementally to a JSONL file (one JSON object per line).

    Args:
        fetch_full_text: If True, visit each speech detail page and retrieve
                         the full text. Significantly increases runtime.
        max_pages:       Maximum number of listing pages to paginate through.
        output_path:     Path to the output JSONL file.

    Returns:
        Total number of speeches written.
    """
    session = requests.Session()
    logger.info(f"Starting scrape for {who} speeches …")

    # Step 1 – collect all entries from the listing pages
    entries = collect_all_entries(session, max_pages=max_pages)
    logger.info("Found %d unique speeches across listing pages.", len(entries))

    # Step 2 – fetch full text and write each speech as a JSONL line
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for entry in tqdm(entries, desc="Fetching speeches", unit="speech"):
            full_text = None
            if fetch_full_text:
                full_text = fetch_speech_text(entry["url"], session)
                time.sleep(REQUEST_DELAY)

            speech = Speech(
                title=entry["title"],
                url=entry["url"],
                date=entry["date"],
                speech_type=entry["speech_type"],
                full_text=full_text,
            )
            fh.write(json.dumps(asdict(speech), ensure_ascii=False) + "\n")
            count += 1

    logger.info("Total speeches written: %d → %s", count, output_path)
    return count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape speeches from vie-publique.fr")
    parser.add_argument(
        "--no-full-text",
        action="store_true",
        help="Skip fetching the full text of each speech (faster, metadata only)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Maximum number of listing pages to paginate through (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="chirac_speeches.jsonl",
        help="Output JSONL file path (default: chirac_speeches.jsonl)",
    )
    parser.add_argument(
        "--who",
        default="Jacques Chirac",
        help="Who to scrape speeches for (default: Jacques Chirac)",
    )
    args = parser.parse_args()

    total = scrape_all_speeches(
        fetch_full_text=not args.no_full_text,
        max_pages=args.max_pages,
        output_path=Path(args.output),
        who=args.who,
    )
    print(f"\nDone. {total} speeches written to '{args.output}'.")
