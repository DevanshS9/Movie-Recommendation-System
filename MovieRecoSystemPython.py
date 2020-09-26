#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process


# In[3]:


df = pd.read_csv('ratings.csv')


# In[4]:


df.head()


# In[5]:


movie_titles = pd.read_csv("movies.csv")
movie_titles.head()


# In[6]:


df = pd.merge(df,movie_titles,on='movieId')
df.head()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


sns.set_style('white')


# In[9]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head(10)


# In[10]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head(10)


# In[11]:


ratings =pd.DataFrame(df.groupby('title')['rating'].mean())


# In[12]:


ratings.head()


# In[13]:


ratings['rating_numbers'] = pd.DataFrame(df.groupby('title')['rating'].count())


# In[14]:


ratings.head()


# In[15]:


ratings['rating_numbers'].hist(bins=70)


# In[16]:


ratings['rating'].hist(bins=70)


# In[17]:


sns.jointplot(x='rating', y='rating_numbers', data=ratings, alpha=0.5)


# In[18]:


moviemat = df.pivot_table(index='userId', columns='title', values='rating')
moviemat.head()


# In[19]:


ratings.sort_values('rating_numbers', ascending=False).head(10)


# In[20]:


genres = movie_titles['genres']


# In[21]:


genre_list = ""
for index,row in movie_titles.iterrows():
        genre_list += row.genres + "|"
#split the string into a list of values
genre_list_split = genre_list.split('|')
#de-duplicate values
new_list = list(set(genre_list_split))
#remove the value that is blank
new_list.remove('')
#inspect list of genres
new_list


# In[22]:


#Enriching the movies dataset by adding the various genres columns.
movies_with_genres = movie_titles.copy()

for genre in new_list :
    movies_with_genres[genre] = movies_with_genres.apply(lambda _:int(genre in _.genres), axis = 1)


# In[23]:


movies_with_genres.head()


# In[24]:


pulpfiction_user_ratings = moviemat['Pulp Fiction (1994)']
forestgump_user_ratings = moviemat['Forrest Gump (1994)']


# In[25]:


pulpfiction_user_ratings.head()


# In[26]:


similar_to_pulpfiction = moviemat.corrwith(pulpfiction_user_ratings)
similar_to_pulpfiction.head()


# In[27]:


corr_pulpfiction = pd.DataFrame(similar_to_pulpfiction, columns=['Correlation'])
corr_pulpfiction.dropna(inplace=True)


# In[28]:


corr_pulpfiction.head()


# In[29]:


corr_pulpfiction.sort_values('Correlation', ascending=False).head(10)


# In[30]:


corr_pulpfiction = corr_pulpfiction.join(ratings['rating_numbers'], how='left', lsuffix='_left', rsuffix='_right')
corr_pulpfiction.head()


# In[31]:


corr_pulpfiction[corr_pulpfiction['rating_numbers']>100].sort_values('Correlation', ascending=False).head()


# In[32]:


similar_to_forestgump = moviemat.corrwith(forestgump_user_ratings)
similar_to_forestgump.head()


# In[33]:


corr_forestgump = pd.DataFrame(similar_to_forestgump, columns=['Correlation'])
corr_forestgump.head()


# In[34]:


corr_forestgump.dropna(inplace=True)


# In[35]:


corr_forestgump = corr_forestgump.join(ratings['rating_numbers'], how='left')
corr_forestgump.head()


# In[36]:


corr_forestgump[corr_forestgump['rating_numbers']>100].sort_values('Correlation', ascending=False).head()


# In[37]:


movies='movies.csv'
ratings='ratings.csv'

df_movies=pd.read_csv(movies, usecols=['movieId','title'], dtype={'movieId':'int32','title':'str'})
df_ratings=pd.read_csv(ratings, usecols=['userId','movieId','rating'],dtype={'userId':'int32','movieId':'int32','rating':'float32'})


# In[38]:


movies_users=df_ratings.pivot(index='movieId', columns='userId',values='rating').fillna(0)
mat_movies_users=csr_matrix(movies_users.values)


# #### Algorithm Applied

# In[39]:


model_knn= NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)


# In[40]:


model_knn.fit(mat_movies_users)


# In[41]:


# Recommender(movie_name) => List of Movies recommended

def recommender(movie_name, data,model, n_recommendations ):
    model.fit(data)
    idx=process.extractOne(movie_name, df_movies['title'])[2]
    print('Movie Selected: ',df_movies['title'][idx], 'Index: ',idx)
    print('Searching for recommendations.....')
    distances, indices=model.kneighbors(data[idx], n_neighbors=n_recommendations)
    for i in indices:
        print(df_movies['title'][i].where(i!=idx))
    
recommender('iron man', mat_movies_users, model_knn,20)


# In[42]:


from lightfm.datasets import fetch_movielens
from lightfm import LightFM


# In[43]:


#fetch data from model
data = fetch_movielens(min_rating = 4.0)

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

#create model
model = LightFM(loss = 'warp')

#train mode
model.fit(data['train'], epochs=30, num_threads=2)

#recommender fucntion
def sample_recommendation(model, data, user_ids):
    #number of users and movies in training data
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
    	#movies they already like
        known_liked_movies = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        #sort them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]
        #print out the results
        print("User %s" % user_id)
        print("     Known Liked movies:")

        for x in known_liked_movies[:3]:
            print("        %s" % x)

        print("     Recommended Movies:")

        for x in top_items[:5]:
            print("        %s" % x)
            
sample_recommendation(model, data, [1,2,3,5,25, 451])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




