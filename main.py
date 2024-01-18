from flask import Flask, request, jsonify
from atlas import execute_vector_search, update_plot
from sagemaker import create_embedding


app = Flask(__name__)


@app.route("/movies/<title>", methods=["PUT"])
def update_movie(title: str):
    try:
        request_json = request.get_json()
        plot = request_json["plot"]
        updated_movie = update_plot(title, plot)

        if updated_movie:
            return jsonify(
                {
                    "message": "Movie updated successfully",
                    "updated_movie": updated_movie,
                }
            )
        else:
            return jsonify({"error": f"Movie with title {title} not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/movies/search", methods=["POST"])
def search_movies():
    try:
        request_json = request.get_json()
        query = request_json["query"]
        embedding = create_embedding(query)

        search_results = execute_vector_search(embedding)

        return jsonify(
            {
                "message": "Movies searched successfully",
                "search_results": list(search_results),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # app.run(debug=True)
    embedding = create_embedding("foo")
    print(embedding)
