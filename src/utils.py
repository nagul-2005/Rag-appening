import math
import logging
import urllib.request
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)


def download_pdf(url: str, dest_path: Union[str, Path]) -> Path:
    """Download the eBook PDF if it doesn't already exist locally."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    logger.info(f"Downloading PDF from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        while chunk := resp.read(8192):
            f.write(chunk)

    logger.info("PDF download complete.")
    return dest_path


def distance_to_score(distance: float) -> float:
    """
    Convert vector distance (where 0 is identical) to a similarity score between 0 and 1.
    Formula: 1 / (1 + distance)
    """
    if math.isnan(distance) or math.isinf(distance):
        return 0.0
    d = max(0.0, float(distance))
    return round(1.0 / (1.0 + d), 4)


def compute_confidence(scores: List[float]) -> float:
    """
    Calculate an overall confidence score for the retrieved chunks.
    Gives 60% weight to the top matching chunk and 40% to the average.
    """
    if not scores:
        return 0.0
    valid = [s for s in scores if s > 0.0]
    if not valid:
        return 0.0

    top = max(valid)
    avg = sum(valid) / len(valid)
    return round((0.6 * top) + (0.4 * avg), 2)
