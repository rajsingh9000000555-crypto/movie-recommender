from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .recommender import Recommender
import os
from urllib.parse import quote
from urllib.request import urlopen
from functools import lru_cache
import json
import ssl

app = FastAPI(title="CineMatch AI API", version="1.0.0", description="Movie discovery and explainable recommendations")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = Recommender()
watchlists = {}
ratings = {}
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")

@lru_cache(maxsize=100)
def fetch_omdb_poster(title: str, year: int):
    if not OMDB_API_KEY:
        return None
    url = "https://www.omdbapi.com/?apikey=%s&t=%s&y=%s" % (
        quote(OMDB_API_KEY),
        quote(title),
        quote(str(year)),
    )
    try:
        context = ssl._create_unverified_context()
        with urlopen(url, timeout=8, context=context) as response:
            data = json.loads(response.read().decode("utf-8"))
        poster = data.get("Poster")
        return poster if poster and poster != "N/A" else None
    except Exception:
        return None

@app.get("/api/movies/{movie_id}/poster")
def movie_poster(movie_id: int):
    movie = engine.get(movie_id)
    if movie is None:
        raise HTTPException(404, "Movie not found")
    poster = fetch_omdb_poster(str(movie["title"]), int(movie["year"]))
    return {"poster": poster}
class RatingPayload(BaseModel):
    movie_id: int
    rating: float = Field(ge=0.5, le=5.0)
@app.get("/")
def root(): return {"name":"CineMatch AI", "status":"ok", "docs":"/docs"}
@app.get("/api/movies")
def list_movies(q: str = "", genre: str = "", limit: int = Query(50, ge=1, le=100)):
    return engine.search(q, genre, limit)
@app.get("/api/movies/trending")
def trending(): return engine.trending()
@app.get("/api/genres")
def genres():
    vals=set()
    for g in engine.df.genres:
        vals.update(g.split("|"))
    return sorted(vals)
@app.get("/api/movies/{movie_id}")
def movie_detail(movie_id: int):
    movie=engine.get(movie_id)
    if movie is None: raise HTTPException(404, "Movie not found")
    return movie
@app.get("/api/movies/{movie_id}/similar")
def similar(movie_id: int, limit: int = Query(8, ge=1, le=20)):
    if engine.get(movie_id) is None: raise HTTPException(404, "Movie not found")
    return engine.similar(movie_id, limit)
@app.get("/api/recommendations/{movie_id}")
def recommend(movie_id: int, limit: int = Query(8, ge=1, le=20)):
    if engine.get(movie_id) is None: raise HTTPException(404, "Movie not found")
    return {"source":engine.get(movie_id), "recommendations":engine.similar(movie_id, limit), "method":"TF-IDF + cosine similarity"}
@app.get("/api/watchlist/{user_id}")
def get_watchlist(user_id: str):
    return [engine.get(mid) for mid in watchlists.get(user_id, []) if engine.get(mid)]
@app.post("/api/watchlist/{user_id}/{movie_id}")
def add_watchlist(user_id: str, movie_id: int):
    if engine.get(movie_id) is None: raise HTTPException(404, "Movie not found")
    saved=watchlists.setdefault(user_id, [])
    if movie_id not in saved: saved.append(movie_id)
    return {"saved":True, "count":len(saved)}
@app.delete("/api/watchlist/{user_id}/{movie_id}")
def remove_watchlist(user_id: str, movie_id: int):
    saved=watchlists.setdefault(user_id, [])
    if movie_id in saved: saved.remove(movie_id)
    return {"saved":False, "count":len(saved)}
@app.post("/api/ratings/{user_id}")
def rate(user_id: str, payload: RatingPayload):
    if engine.get(payload.movie_id) is None: raise HTTPException(404, "Movie not found")
    ratings.setdefault(user_id, {})[payload.movie_id]=payload.rating
    return {"saved":True,"movie_id":payload.movie_id,"rating":payload.rating}
