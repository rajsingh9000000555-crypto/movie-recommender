# CineMatch AI — Movie Discovery & Recommendation System

A BTech second-year portfolio project that combines a movie discovery UI with a simple, explainable machine-learning recommendation engine.

## What it includes
- Content-based movie recommendations using TF-IDF features and cosine similarity
- Popularity ranking baseline
- Search by title, overview, or director; genre filtering
- Similar-movie explanations based on overlapping genres/themes
- Watchlist saved in browser local storage
- Demo movie-rating endpoint
- FastAPI backend with Swagger docs
- Responsive frontend with a dark cinematic design

> This is a learning/demo application. Watchlist and ratings are stored in memory on the API or browser local storage, not in a persistent user database. The included movie list is a small sample catalog. Posters are external images and may not all load; the UI has a built-in poster fallback.

## Tech stack
Python, FastAPI, pandas, scikit-learn, TF-IDF, cosine similarity, HTML/CSS/JavaScript.

## Run locally
### 1. Start backend
Windows PowerShell / CMD:
```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Linux/macOS:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API docs: http://127.0.0.1:8000/docs

### 2. Start frontend
Open a second terminal from the project root:
```bash
python -m http.server 5173 --directory frontend
```
Open http://127.0.0.1:5173

The frontend calls the backend at `http://127.0.0.1:8000`. If you change ports, update `API` in `frontend/app.js`.

## ML pipeline
1. Combine genres, overview, director and keywords into a text feature.
2. Fit `TfidfVectorizer` with English stop words and 1–2 grams.
3. Compute cosine similarity between each movie feature vector.
4. Given a selected film, rank all other films by similarity and return top K.
5. Return an explanation based on shared genres.

The trending endpoint uses a simple rating/vote-weighted score. This is a baseline, not a production ranking formula.

## API endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Health check |
| GET | `/api/movies?q=&genre=&limit=` | Catalog search/filter |
| GET | `/api/movies/trending` | Popularity-ranked catalog |
| GET | `/api/genres` | Available genres |
| GET | `/api/movies/{movie_id}` | Movie detail |
| GET | `/api/movies/{movie_id}/similar?limit=` | Similar movies |
| GET | `/api/recommendations/{movie_id}?limit=` | Recommendation payload + explanation |
| GET | `/api/watchlist/{user_id}` | Demo watchlist retrieval |
| POST | `/api/watchlist/{user_id}/{movie_id}` | Add movie |
| DELETE | `/api/watchlist/{user_id}/{movie_id}` | Remove movie |
| POST | `/api/ratings/{user_id}` | Save demo rating `{ "movie_id": 1, "rating": 4.5 }` |

## Project structure
```text
CineMatch-AI/
├── backend/
│   ├── app/main.py           # API endpoints
│   ├── app/recommender.py   # TF-IDF & similarity logic
│   ├── data/movies.csv     # Sample catalog
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── README.md
```

## Suggested next upgrades
- Persist accounts, ratings and watchlists with SQLite/PostgreSQL
- Implement collaborative filtering and hybrid ranking
- Add offline evaluation (Precision@K, Recall@K, NDCG@K, catalog coverage)
- Add JWT authentication, automated tests, Docker, CI and deployment
- Replace sample CSV with a properly licensed larger dataset; respect each dataset's license

## Viva notes
**TF-IDF:** converts text into weighted numeric features; terms that are common across all films receive less weight than terms that distinguish a movie.

**Cosine similarity:** measures the angle between two feature vectors; values closer to 1 indicate more similar text-feature direction.

**Content-based filtering:** recommends items similar to an item a user liked, based on item attributes. It does not need ratings from other users.

**Cold start:** popularity-ranked films are available as a fallback for a new visitor without preference history.
