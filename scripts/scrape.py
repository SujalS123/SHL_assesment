#!/usr/bin/env python3
# scripts/scrape.py — Scrapes the SHL product catalog and saves catalog.json
#
# Run ONCE offline: python scripts/scrape.py
# Output: data/catalog.json
#
# Scrapes Individual Test Solutions only (Pre-packaged Job Solutions are out of scope).
# Each entry: { name, url, test_type, description, competencies }

import json
import time
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "catalog.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def get_catalog_page(page: int = 1, per_page: int = 12) -> BeautifulSoup:
    """Fetch one page of the catalog listing."""
    params = {
        "type": "1",          # Individual Test Solutions filter
        "start": (page - 1) * per_page,
    }
    resp = requests.get(CATALOG_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_all_listing_links() -> list[dict]:
    """
    Paginate through catalog listing and collect all product links.
    Returns list of {name, url} dicts.
    """
    items = []
    page = 1
    while True:
        print(f"  Fetching listing page {page}...")
        soup = get_catalog_page(page)

        # Product cards are in <div class="product-catalogue-training-calendar__row">
        cards = soup.select(".product-catalogue-training-calendar__row")
        if not cards:
            break

        for card in cards:
            link_tag = card.select_one("a[href]")
            if link_tag:
                name = link_tag.get_text(strip=True)
                url = BASE_URL + link_tag["href"] if link_tag["href"].startswith("/") else link_tag["href"]
                items.append({"name": name, "url": url})

        # Check for a next page
        next_btn = soup.select_one("a[rel='next']")
        if not next_btn:
            break
        page += 1
        time.sleep(0.5)   # be polite

    return items


def scrape_product_page(url: str) -> dict:
    """
    Scrape an individual product page for description, test_type, competencies.
    Returns a dict with those fields (empty strings/lists if not found).
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Description — first substantial paragraph in the product content area
        description = ""
        desc_tag = soup.select_one(".product-catalogue__description, .solution-description, article p")
        if desc_tag:
            description = desc_tag.get_text(strip=True)[:500]

        # Test type — look for labels like "Ability", "Personality", "Knowledge", etc.
        test_type = _extract_test_type(soup)

        # Competencies — bullet lists or tags associated with the product
        competencies = _extract_competencies(soup)

        return {
            "description": description,
            "test_type": test_type,
            "competencies": competencies,
        }
    except Exception as e:
        print(f"    WARNING: failed to scrape {url}: {e}")
        return {"description": "", "test_type": "U", "competencies": []}


# Map of keywords found in test-type labels → single-letter code used in the spec
_TYPE_MAP = {
    "ability": "A",
    "aptitude": "A",
    "cognitive": "A",
    "personality": "P",
    "behavioural": "P",
    "behavioral": "P",
    "knowledge": "K",
    "skills": "K",
    "situational": "S",
    "judgment": "S",
    "biodata": "B",
    "motivation": "M",
    "competency": "C",
}


def _extract_test_type(soup: BeautifulSoup) -> str:
    """Guess the single-letter test type code from the product page."""
    text = soup.get_text(" ", strip=True).lower()
    for keyword, code in _TYPE_MAP.items():
        if keyword in text:
            return code
    return "U"   # Unknown


def _extract_competencies(soup: BeautifulSoup) -> list[str]:
    """Extract competency labels (tags, bullet items, etc.)."""
    comps = []
    # Try <ul> items inside a competencies section
    for li in soup.select(".competencies li, .tags li, .skills li"):
        text = li.get_text(strip=True)
        if text:
            comps.append(text)
    return comps[:10]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Step 1: collecting catalog listing links...")
    items = get_all_listing_links()
    print(f"  Found {len(items)} products.")

    print("Step 2: scraping individual product pages...")
    catalog = []
    for i, item in enumerate(items):
        print(f"  [{i+1}/{len(items)}] {item['name']}")
        details = scrape_product_page(item["url"])
        entry = {**item, **details}
        catalog.append(entry)
        time.sleep(0.3)

    print(f"Step 3: saving {len(catalog)} entries to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(catalog, f, indent=2)

    print("Done! catalog.json is ready.")


if __name__ == "__main__":
    main()
