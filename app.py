from flask import Flask, request, jsonify
from models.collaborative import CollaborativeFiltering
from models.content_based import ContentBasedFiltering

app = Flask(__name__)

# Initialize recommendation systems
collab_filter = CollaborativeFiltering('data/ratings.csv', 'data/movies.csv')
content_filter = ContentBasedFiltering('data/movies.csv')

@app.route('/recommend/collaborative/<int:user_id>', methods=['GET'])
def collaborative_recommend(user_id):
    try:
        n = int(request.args.get('n', 5))
        recommendations = collab_filter.recommend_for_user(user_id, n)
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations,
            "type": "collaborative_filtering"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/recommend/content-based/<int:movie_id>', methods=['GET'])
def content_based_recommend(movie_id):
    try:
        n = int(request.args.get('n', 5))
        recommendations = content_filter.recommend_similar_movies(movie_id, n)
        return jsonify({
            "movie_id": movie_id,
            "recommendations": recommendations,
            "type": "content_based_filtering"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/')
def home():
    return """
    <h1>Movie Recommendation System</h1>
    <p>Available endpoints:</p>
    <ul>
        <li>Collaborative filtering: GET /recommend/collaborative/&lt;user_id&gt;?n=5</li>
        <li>Content-based filtering: GET /recommend/content-based/&lt;movie_id&gt;?n=5</li>
    </ul>
    """

if __name__ == '__main__':
    app.run(debug=True)
