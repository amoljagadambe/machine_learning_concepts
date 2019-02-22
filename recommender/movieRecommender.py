import pandas as pd
import os
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

filepath = os.getcwd()
df = pd.read_csv(filepath +'/movie_rating.csv')

""" link to download dataset
https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7 """

df = df[['Title','Genre','Director','Actors','Plot']]

df['Actors'] = df['Actors'].map(lambda x: x.split(',')[:3])
df['Genre'] = df['Genre'].map(lambda x: x.lower().split(','))
df['Director'] = df['Director'].map(lambda x: x.split(' '))

for index, row in df.iterrows():
    row['Actors'] = [x.lower().replace(' ','') for x in row['Actors']]
    row['Director'] = ''.join(row['Director']).lower()
#print(df['Director'])

# initializing the new column
df['Key_words'] = ""

for index, row in df.iterrows():
    plot = row['Plot']

    # instantiating Rake, by default it uses english stopwords from NLTK
    # and discards all puntuation characters as well
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(plot)

    # getting the dictionary whith key words as keys and their scores as values
    key_words_dict_scores = r.get_word_degrees()

    # assigning the key words to the new column for the corresponding movie
    row['Key_words'] = list(key_words_dict_scores.keys())

#dropping the Plot column
df.drop(columns=['Plot'], inplace=True)


df.set_index('Title', inplace = True)

df['bag_of_words'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if col != 'Director':
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['bag_of_words'] = words

df.drop(columns=['Genre','Director','Actors','Key_words'], inplace=True)
#print(df.head())

count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])

indices = pd.Series(df.index)
#print(indices[:5])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
#print(cosine_sim)


# function that takes in movie title as input and returns the top 10 recommended movies
def recommendations(title, cosine_sim=cosine_sim):
    recommended_movies = []

    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])

    return print(recommended_movies)

if __name__=='__main__':

    name  = input('Enter the Movie name to obtain the suggestion: ')
    recommendations(name)