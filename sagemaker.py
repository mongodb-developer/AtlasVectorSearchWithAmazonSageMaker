import os
from typing import Optional
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_SERVICE = os.environ.get("EMBEDDING_SERVICE")


def create_embedding(plot: str) -> Optional[float]:
    encoded_plot = quote(plot)
    embedding_url = f"{EMBEDDING_SERVICE}?query={encoded_plot}"

    embedding_response = requests.get(embedding_url)
    embedding_vector = embedding_response.json()["embedding"]

    return embedding_vector
