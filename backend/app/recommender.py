from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA = Path(__file__).resolve().parents[1] / "data" / "movies.csv"
class Recommender:
    def __init__(self):
        self.df = pd.read_csv(DATA).fillna("")
        self.df["features"] = (self.df["genres"].astype(str).str.replace("|", " ", regex=False) + " " +
            self.df["overview"].astype(str) + " " + self.df["director"].astype(str) + " " + self.df["keywords"].astype(str))
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000)
        self.matrix = self.vectorizer.fit_transform(self.df["features"])
        self.similarity = cosine_similarity(self.matrix)
    def movies(self):
        return self.df.drop(columns=["features"]).to_dict(orient="records")
    def get(self, movie_id):
        rows = self.df[self.df.id == int(movie_id)]
        if rows.empty: return None
        return rows.iloc[0].drop(labels=["features"]).to_dict()
    def similar(self, movie_id, limit=8):
        matches = self.df.index[self.df.id == int(movie_id)].tolist()
        if not matches: return []
        idx = matches[0]
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted((x for x in scores if x[0] != idx), key=lambda x: x[1], reverse=True)[:limit]
        output = []
        source_genres = set(self.df.iloc[idx].genres.split("|"))
        for i, score in scores:
            movie = self.df.iloc[i].drop(labels=["features"]).to_dict()
            shared = source_genres.intersection(set(movie["genres"].split("|")))
            movie["similarity"] = round(float(score), 3)
            movie["reason"] = "Shared themes and genres: " + (", ".join(sorted(shared)) if shared else "related movie metadata")
            output.append(movie)
        return output
    def trending(self, limit=10):
        df = self.df.copy()
        df["pop_score"] = df.rating.astype(float) * (df.votes.astype(float).clip(lower=1).apply(lambda n: n/(n+50000)))
        return df.sort_values("pop_score", ascending=False).head(limit).drop(columns=["features", "pop_score"]).to_dict(orient="records")
    def search(self, query="", genre="", limit=50):
        df = self.df.copy()
        if query.strip():
            q = query.lower()
            df = df[df.title.str.lower().str.contains(q, regex=False) | df.overview.str.lower().str.contains(q, regex=False) | df.director.str.lower().str.contains(q, regex=False)]
        if genre and genre.lower() != "all":
            df = df[df.genres.str.lower().str.contains(genre.lower(), regex=False)]
        return df.head(limit).drop(columns=["features"]).to_dict(orient="records")
