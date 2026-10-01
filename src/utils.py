import math
import logging
from pathlib import Path
import urllib.request
from typing import List, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_pdf(url: str, dest_path: Union[str, Path]) -> Path:
    """
    Downloads a PDF document from a given URL if it does not already exist at dest_path.

    Args:
        url: Direct HTTP/HTTPS download link.
        dest_path: Destination file path on local disk.

    Returns:
        Path object pointing to the downloaded PDF file.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"PDF already exists at {dest_path} ({dest_path.stat().st_size} bytes). Skipping download.")
        return dest_path

    logger.info(f"Downloading PDF from {url} to {dest_path}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request) as response, open(dest_path, "wb") as out_file:
            content_length = response.getheader("Content-Length")
            total_size = int(content_length) if content_length else 0
            downloaded = 0
            block_size = 8192

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)
                if total_size > 0 and downloaded % (block_size * 500) == 0:
                    percent = (downloaded / total_size) * 100
                    logger.info(f"Download progress: {percent:.1f}% ({downloaded}/{total_size} bytes)")

        logger.info(f"Successfully downloaded PDF to {dest_path} ({dest_path.stat().st_size} bytes).")
        return dest_path

    except Exception as e:
        logger.error(f"Failed to download PDF from {url}: {str(e)}")
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Error downloading PDF file: {e}") from e


def normalize_distance_to_score(raw_score: float, metric: str = "cosine") -> float:
    """
    Normalizes distance / similarity metrics into a normalized relevance score between 0.0 and 1.0.

    Args:
        raw_score: Raw distance or similarity value returned by vector store.
        metric: Distance metric used ('cosine', 'l2', or 'relevance').

    Returns:
        Float relevance score between 0.0 and 1.0.
    """
    if math.isnan(raw_score) or math.isinf(raw_score):
        return 0.0

    # If raw_score is already a valid relevance ratio between 0 and 1
    if metric == "relevance" and 0.0 <= raw_score <= 1.0:
        return round(float(raw_score), 4)

    # Convert distance metrics d >= 0 to similarity score s in (0, 1]
    # Formula: s = 1 / (1 + max(0, distance))
    d = max(0.0, float(raw_score))
    score = 1.0 / (1.0 + d)
    return round(score, 4)



def calculate_confidence_score(relevance_scores: List[float]) -> float:
    """
    Computes an overall grounded confidence score from a list of chunk relevance scores.

    Formula combines the top relevance match with the average of top chunks to reward
    both high peak precision and broad context coverage.

    Args:
        relevance_scores: List of individual chunk relevance scores [0.0 to 1.0].

    Returns:
        Float confidence score bounded strictly between 0.0 and 1.0 rounded to 2 decimals.
    """
    if not relevance_scores:
        return 0.0

    valid_scores = [s for s in relevance_scores if s > 0.0]
    if not valid_scores:
        return 0.0

    top_score = max(valid_scores)
    mean_score = sum(valid_scores) / len(valid_scores)

    # 60% weight on best match, 40% weight on average context relevance
    confidence = (0.60 * top_score) + (0.40 * mean_score)
    return round(max(0.0, min(1.0, confidence)), 2)
