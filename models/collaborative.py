import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, ratings_path, movies_path):
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        self.user_movie_matrix = None
        self.user_similarity = None
        self._prepare_data()

    def _prepare_data(self):
        # Create user-item matrix
        self.user_movie_matrix = self.ratings.pivot_table(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)
        
        # Calculate user similarity
        self.user_similarity = cosine_similarity(self.user_movie_matrix)
        self.user_similarity = pd.DataFrame(
            self.user_similarity, 
            index=self.user_movie_matrix.index, 
            columns=self.user_movie_matrix.index
        )

    def recommend_for_user(self, user_id, n=5):
        # Find similar users
        similar_users = self.user_similarity[user_id].sort_values(ascending=False)[1:n+1]
        
        # Get movies rated by similar users
        similar_users_ratings = self.user_movie_matrix.loc[similar_users.index]
        weighted_ratings = similar_users_ratings.mul(similar_users, axis=0)
        recommended_movies = weighted_ratings.sum().sort_values(ascending=False)
        
        # Filter out movies already rated by the user
        user_rated = set(self.ratings[self.ratings['userId'] == user_id]['movieId'])
        recommended_movies = recommended_movies[~recommended_movies.index.isin(user_rated)]
        
        # Get movie details
        recommendations = self.movies[self.movies['movieId'].isin(recommended_movies.index)]
        recommendations = recommendations.merge(
            pd.DataFrame({'movieId': recommended_movies.index, 'score': recommended_movies.values}),
            on='movieId'
        ).sort_values('score', ascending=False).head(n)
        
        return recommendations.to_dict('records')
