#1. Import the required modules.
from bs4 import BeautifulSoup
import requests
import pandas as pd

#2. Access the HTML content from the IMDb Top 250 movies page
url = 'https://www.imdb.com/chart/top/'
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

#3. Extract movie details using HTML tags, each li tag represents a movie block containing title, year, and rating details.
movies = soup.select("li.ipc-metadata-list-summary-item")

#4. Create a list to store movie data
movie_data = []

movie_data = []

for movie in movies:
    title = movie.select_one("h3.ipc-title__text").text.strip()
    year = movie.select_one("span.cli-title-metadata-item").text.strip()
    rating_tag = movie.select_one("span.ipc-rating-star--rating")
    rating = rating_tag.text.strip() if rating_tag else "N/A"
    
    movie_data.append({
        "Title": title,
        "Year": year,
        "Rating": rating
    })

    
#5.Display the extracted data
for movie in movie_data:
    print(f"{movie ['Title']} ({movie['Year']}) - Rating: {movie['Rating']}")

#6. Save the data into a CSV file
df = pd.DataFrame(movie_data)
df.to_csv("imdb_top_250_movies.csv", index=False)
print("IMDb data saved successfully to imdb_250_movies.csv!")
