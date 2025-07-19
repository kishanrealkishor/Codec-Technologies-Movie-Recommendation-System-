import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedFiltering:
    def __init__(self, movies_path):
        self.movies = pd.read_csv(movies_path)
        self.tfidf_matrix = None
        self.cosine_sim = None
        self._prepare_data()

    def _prepare_data(self):
        # Create TF-IDF matrix for movie genres
        tfidf = TfidfVectorizer(stop_words='english')
        self.movies['genres'] = self.movies['genres'].fillna('')
        self.tfidf_matrix = tfidf.fit_transform(self.movies['genres'])
        
        # Compute cosine similarity matrix
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def recommend_similar_movies(self, movie_id, n=5):
        # Get the index of the movie
        idx = self.movies[self.movies['movieId'] == movie_id].index[0]
        
        # Get pairwise similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort movies by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top n similar movies (skip the first one as it's the same movie)
        sim_scores = sim_scores[1:n+1]
        
        # Get movie indices and scores
        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        # Return top n similar movies
        recommendations = self.movies.iloc[movie_indices].copy()
        recommendations['score'] = scores
        return recommendations.to_dict('records')
