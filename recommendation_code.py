import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.models import Model
from keras import regularizers

# Read in the ratings and movies data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge the ratings and movies data using the movie_id as the key
data = pd.merge(ratings, movies, on='movieId')
# Create a new table with selected columns
new_table = pd.DataFrame(data[['userId', 'movieId', 'rating','timestamp']])

user_ids=ratings["userId"].unique().tolist()

userencoded= {x:i for i,x in enumerate(user_ids)}
user_rev={i:x for i,x in enumerate(user_ids)}


movie_ids= ratings['movieId'].unique().tolist()
moviecoded = {x:i for i,x in enumerate(movie_ids)}
movie_rev = {i:x for i,x in enumerate(movie_ids)}

ratings['user']= ratings['userId'].map(userencoded)
ratings['movie']= ratings['movieId'].map(moviecoded)

num_users= len(userencoded)
num_movies = len(moviecoded)

ratings['rating']=ratings['rating'].values.astype(np.float32)

max_rating=max(ratings['rating'])
min_rating = min(ratings['rating'])


ratings = ratings.sample(frac=1,random_state=42)
x= ratings[['user','movie']].values
y = ratings['rating'].apply(lambda x: (x-min_rating)/(max_rating-min_rating)).values

train_indices=(int(0.9 * ratings.shape[0]))
x_train , xval, y_train, yval=(
    x[:train_indices],x[train_indices:],y[:train_indices],y[train_indices:]
)

embedding_size= 50
user_layer= layers.Input(shape=[1])
user_embedding= layers.Embedding(num_users,embedding_size,embeddings_initializer= "he_normal",embeddings_regularizer= keras.regularizers.l2(1e-6))(user_layer)
user_vector=layers.Flatten()(user_embedding)

movie_layer=layers.Input(shape=[1])
movie_embedding= layers.Embedding(num_movies,embedding_size,embeddings_initializer="he_normal",embeddings_regularizer=keras.regularizers.l2(1e-6))(movie_layer)
movie_vector=layers.Flatten()(movie_embedding)

prod= layers.dot(inputs=[user_vector,movie_vector],axes=1)

dense1=layers.Dense(150,activation='relu',kernel_initializer="he_normal")(prod)
dense2= layers.Dense(50,activation='relu',kernel_initializer="he_normal")(dense1)
dense3= layers.Dense(1,activation='relu')(dense2)
model= Model([user_layer,movie_layer],dense3)
model.compile(optimizer="Adam", loss="mean_squared_error")
history= model.fit([x_train[:,0],x_train[:,1]],y_train,batch_size=64,epochs=2,verbose=1)
#pred=model.predict(([x_train[4:5,0],x_train[4:5,1]]))

use=-1
while use < 0 or use > 610:
    use=int(input("Enter the user ID: "))

movies_watched=ratings[ratings.userId==use]
movies_not_watched= movies[~movies["movieId"].isin(movies_watched.movieId.values)]["movieId"]

movies_not_watched=list(set(movies_not_watched).intersection(set(moviecoded.keys())))

movies_not_watched_index= [[moviecoded.get(x)] for x in movies_not_watched]

user_encoder = userencoded.get(use)
user_movie_array= np.hstack(([[user_encoder]]*len(movies_not_watched),movies_not_watched_index))


predicted_ratings = model.predict([user_movie_array[:, 0], user_movie_array[:, 1]]).flatten()
print(predicted_ratings)
top_rating_indices= predicted_ratings.argsort()[-10:][::-1]
print("Movies with high rating from user:\n")
pre_liked= (movies_watched.sort_values(by="rating",ascending=False).head(10).movieId.values
            )
movie_rows= movies[movies['movieId'].isin(pre_liked)]
for row in movie_rows.itertuples():
    print(row.title,":",row.genres)
    
recommended_movies = [movie_rev.get(movies_not_watched_index[x][0]) for x in top_rating_indices]
print("\n****************TOP 10 Recommended movies for the user ",use,"***********************\n\n\n\n")
print("The top 10 recommended movies are:\n")


recommender=movies[movies["movieId"].isin(recommended_movies)]
for row in recommender.itertuples():
    print(row.title,":",row.genres)