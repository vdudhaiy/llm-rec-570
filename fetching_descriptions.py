import requests
import pandas as pd
import re
import time

with open("tmdb_api_key.txt", "r") as f:
    TMDB_API_KEY = f.read()
DELAY = 1

def get_tmdb_id(title):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    
    time.sleep(DELAY)

    response = requests.get(search_url)
    data = response.json()
    
    if data["results"]:
        return data["results"][0]["id"]  # Return the first movie's ID on tmdb
    else:
        return None

def read_movies():
    movies = []
    with open("movielens-1m/movies.dat", "r") as f:
        data = f.readlines()

    for mov in data:
        l = mov.strip().split("::")
        
        match = re.match(r"^(.*?)\s*(?:\([^)]+\))?\s*\((\d{4})\)$", l[1])

        if match:
            title = match.group(1).strip()
            year = match.group(2).strip()
            movie_dict = {"movieID": l[0], "title":title, "year": year,"genres":l[2]} 
            movies.append(movie_dict)
        else:
            print(f"{l[1]} match NOT FOUND")
    return movies
        
def fetch_description(id):
    fetch_url = f"https://api.themoviedb.org/3/movie/{id}?api_key={TMDB_API_KEY}"
    
    time.sleep(DELAY)

    response = requests.get(fetch_url)
    data = response.json()
    
    return data.get("overview", "NOT FOUND")

def generate_descriptions():
    movies = read_movies()
    print("Read movies.dat...")

    for mov in movies:
        mov_id = get_tmdb_id(mov["title"])
        if mov_id == None:
            print(f"{mov['title']} ID NOT FOUND")
            mov["description"] = "ID NOT FOUND"
            continue
        
        description = fetch_description(mov_id)
        mov["description"] = description

    print("Writing to movies_desc.dat...")

    with open("movielens-1m/movies_desc.dat", "w", encoding="utf-8") as f:
        for mov in movies:
            f.write(mov["movieID"] + "::" + mov["title"] + "::" + mov["year"] + "::" + mov["genres"] + "::" + mov["description"] + "\n")

    return

if __name__=="__main__":
    print("Starting data preprocessing...")
    
    start = time.time()
    generate_descriptions()
    end = time.time()

    print(f"Data Processing took {(end-start):.2f} seconds.")
