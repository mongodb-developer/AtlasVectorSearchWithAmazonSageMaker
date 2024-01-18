import os

import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from pymongo import MongoClient

load_dotenv()

app = Flask(__name__)

# MongoDB
MONGODB_CONNECTION_STRING = os.environ.get('MONGODB_CONNECTION_STRING')
DATABASE_NAME = "sample_mflix"
COLLECTION_NAME = "movies"
VECTOR_SEARCH_INDEX_NAME = "VectorSearchIndex"
EMBEDDING_PATH = "embedding"
mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
database = mongo_client[DATABASE_NAME]
games_collection = database[COLLECTION_NAME]

# AWS
EMBEDDING_SERVICE = os.environ.get('EMBEDDING_SERVICE')
:x

def create_embedding(description: str) -> [float]:
    embedding_url = f"{EMBEDDING_SERVICE}?query={description}"
    embedding_response = requests.get(embedding_url)
    embedding_vector = embedding_response.json()['embedding']
    return embedding_vector


def update_description(name: str, description: str) -> dict:
    embedding = create_embedding(description)

    result = games_collection.find_one_and_update(
        {'name': name},
        {'$set': {'description': description, 'embedding': embedding}},
        return_document=True
    )

    return result


@app.route('/games/<name>', methods=['PUT'])
def update_game(name: str):
    try:
        request_json = request.get_json()
        description = request_json['description']
        updated_game = update_description(name, description)

        if updated_game:
            return jsonify({'message': 'Game updated successfully', 'updated_game': updated_game})
        else:
            return jsonify({'error': f'Game with name {name} not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/games/search', methods=['POST'])
def search_games():
    try:
        request_json = request.get_json()
        query = request_json['query']
        embedding = create_embedding(query)

        vector_search_query = {
            "$vectorSearch": {
                "index": VECTOR_SEARCH_INDEX_NAME,
                "path": EMBEDDING_PATH,
                "queryVector": embedding,
                "numCandidates": 10,
                "limit": 5,
                "filter": {}
            }
        }
        search_results = games_collection.aggregate([vector_search_query])

        return jsonify({'message': 'Games searched successfully', 'search_results': list(search_results)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
