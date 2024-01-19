import os

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from sagemaker import create_embedding

load_dotenv()

MONGODB_CONNECTION_STRING = os.environ.get("MONGODB_CONNECTION_STRING")
DATABASE_NAME = "sample_mflix"
COLLECTION_NAME = "embedded_movies"
VECTOR_SEARCH_INDEX_NAME = "VectorSearchIndex"
EMBEDDING_PATH = "embedding"
mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
database = mongo_client[DATABASE_NAME]
movies_collection = database[COLLECTION_NAME]


def update_plot(title: str, plot: str) -> dict:
    embedding = create_embedding(plot)

    result = movies_collection.find_one_and_update(
        {"title": title},
        {"$set": {"plot": plot, "embedding": embedding}},
        return_document=True,
    )

    return result


def add_missing_embeddings():
    movies_with_a_plot_without_embedding_filter = {
        "$and": [
            {"plot": {"$exists": True, "$ne": ""}},
            {"embedding": {"$exists": False}},
        ]
    }
    only_show_plot_projection = {"plot": 1}

    movies = movies_collection.find(
        movies_with_a_plot_without_embedding_filter,
        only_show_plot_projection,
    )

    movies_to_update = []

    for movie in movies:
        embedding = create_embedding(movie["plot"])
        update_operation = UpdateOne(
            {"_id": movie["_id"]},
            {"$set": {"embedding": embedding}},
        )
        movies_to_update.append(update_operation)

    if movies_to_update:
        result = movies_collection.bulk_write(movies_to_update)
        print(f"Updated {result.modified_count} movies")

    else:
        print("No movies to update")


def execute_vector_search(vector: [float]) -> list[dict]:
    vector_search_query = {
        "$vectorSearch": {
            "index": VECTOR_SEARCH_INDEX_NAME,
            "path": EMBEDDING_PATH,
            "queryVector": vector,
            "numCandidates": 10,
            "limit": 5,
        }
    }
    projection = {"$project": {"_id": 0, "title": 1}}
    results = movies_collection.aggregate([vector_search_query, projection])
    results_list = list(results)

    return results_list
