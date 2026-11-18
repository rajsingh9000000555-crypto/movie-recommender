from app.recommender import Recommender

def test_catalog_has_movies():
    model = Recommender()
    assert len(model.movies()) >= 20

def test_similar_movies_excludes_source():
    model = Recommender()
    results = model.similar(1, 5)
    assert len(results) == 5
    assert all(m["id"] != 1 for m in results)
    assert all("similarity" in m and "reason" in m for m in results)

def test_search_by_title():
    model = Recommender()
    results = model.search("Interstellar")
    assert results and results[0]["title"] == "Interstellar"

def test_genre_filter():
    model = Recommender()
    results = model.search(genre="Fantasy")
    assert results
    assert all("Fantasy" in m["genres"] for m in results)
