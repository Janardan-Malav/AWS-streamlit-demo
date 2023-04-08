#!/usr/bin/env python
# coding: utf-8

# ### Problem statement
# Leveraging the data of Zee, create a Recommender System to show personalized movie recommendations based on ratings given by a user and other users similar to them in order to improve user experience

# In[1]:



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)


# In[2]:


movies = pd.read_fwf('C:/Users/USER/Downloads/zee-movies.dat', encoding='ISO-8859-1')
ratings =pd.read_fwf('C:/Users/USER/Downloads/zee-ratings.dat', encoding='ISO-8859-1')
users = pd.read_fwf('C:/Users/USER/Downloads/zee-users.dat', encoding='ISO-8859-1')


# In[3]:


delimiter ="::"

users = users["UserID::Gender::Age::Occupation::Zip-code"].str.split(delimiter,expand = True)
users.columns = ["UserID","Gender","Age","Occupation","Zipcode"]

users["Age"].replace({"1": "Under 18","18": "18-24","25": "25-34",
                          "35": "35-44","45": "45-49","50": "50-55","56": "56+"},inplace=True)

users["Occupation"] = users["Occupation"].astype(int).replace({0: "other",1: "academic/educator",2: "artist",
                                                               3: "clerical/admin",4: "college/grad student",
                                             5: "customer service",6: "doctor/health care",7: "executive/managerial",
                                             8: "farmer" ,9: "homemaker",10: "K-12 student",11: "lawyer",
                                             12: "programmer",13: "retired",14: "sales/marketing",15: "scientist",
                                             16: "self-employed",17: "technician/engineer",
                                             18: "tradesman/craftsman",19: "unemployed",20: "writer"},
                                            )

delimiter ="::"

ratings = ratings["UserID::MovieID::Rating::Timestamp"].str.split(delimiter,expand = True)
ratings.columns = ["UserID","MovieID","Rating","Timestamp"]


movies.drop(["Unnamed: 1","Unnamed: 2"],axis = 1,inplace=True)



delimiter ="::"

movies = movies["Movie ID::Title::Genres"].str.split(delimiter,expand = True)
movies.columns = ["MovieID","Title","Genres"]


movies.shape,ratings.shape,users.shape


# In[4]:


movies # need to take care of Genres . 


# In[5]:


ratings # need to convert timestamp to hrs. 


# In[6]:


users


# In[7]:


# taking out the release year from the title column from movie table :

movies["Release_year"] = movies["Title"].str.extract('^(.+)\s\(([0-9]*)\)$',expand = True)[1]
movies["Title"] = movies["Title"].str.split("(").apply(lambda x:x[0])


# Converting timestamp to hours 

from datetime import datetime
ratings["Watch_Hour"] =ratings["Timestamp"].apply(lambda x:datetime.fromtimestamp(int(x)).hour)
ratings.drop(["Timestamp"],axis = 1,inplace=True)


# In[8]:


movies.shape,ratings.shape,users.shape


# #### Merging all the tables into one data frame : 

# In[9]:


df = users.merge(movies.merge(ratings,on="MovieID",how="outer"),on="UserID",how="outer")


# In[10]:


df.shape


# In[11]:


df


# In[12]:


df_ = df.copy()


# In[13]:


df_.dropna(inplace=True)


# In[14]:


df_.info()


# In[15]:


df_['Release_year']=df_['Release_year'].astype('int32')
df_['Rating']=df_['Rating'].astype('int32')


# In[16]:


bins = [1919, 1929, 1939, 1949, 1959, 1969, 1979, 1989, 2000]
labels = ['20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s']

df_["Released_In"] =  pd.cut(df_['Release_year'], bins=bins, labels=labels)


# In[17]:


import seaborn as sns


# ## Average user rating distribution :

# In[18]:


sns.histplot(df_[['UserID','Rating']].groupby('UserID').mean()["Rating"])
plt.show()
# average ratings given by each user distribution


# In[19]:


sns.histplot(df_[['MovieID','Rating']].groupby('MovieID').mean()["Rating"])
plt.show()
# average rating , that each movie has receieved by users . 


# In[20]:


df_["MovieID"].nunique()


# In[21]:


movies_per_decade = df_[['MovieID','Released_In']].groupby('Released_In').nunique()
movies_per_decade["% of all Movies"] = (movies_per_decade["MovieID"]/(df_["MovieID"].nunique())) * 100
movies_per_decade


# In[22]:


sns.barplot(movies_per_decade.index,
            movies_per_decade["% of all Movies"])
plt.show()


# In[23]:


m = movies[["MovieID","Title","Genres"]]
m["Genres"] = m["Genres"].str.split("|")
m = m.explode("Genres")
m["Genres"] = m["Genres"].replace({"":"Other","Horro":"Horror","Sci-":"Sci-Fi","Sci":"Sci-Fi","Sci-F":"Sci-Fi","Dr":"Drama","Documenta":"Documentary",
                     "Wester":"Western","Fant":"Fantasy","Chil":"Children's","R":"Romance","D":"Drama","Rom":"Romance","Animati":"Animation","Childr":"Children's","Childre":"Children's",
                     "Fantas":"Fantasy","Come":"Comedy","Dram":"Drama","S":"Sci-Fi","Roma":"Romance","A":"Adventure","Children":"Children's","Adventu":"Adventure","Adv":"Adventure",
                      "Wa":"War","Thrille"  :"Thriller","Com":"Comedy","Comed":"Comedy","Acti":"Action","Advent":"Adventure","Adventur":"Adventure","Thri":"Thriller",
                        "Chi":"Children's","Ro":"Romance","F":"Fantasy","We":"Western","Documen":"Documentary","Music":"Musical","Children":"Children's" ,"Horr":"Horror",
                     "Children'":"Children's","Roman":"Romance","Docu":"Documentary","Th":"Thriller","Document":"Documentary"
                    })

m = m.pivot_table(values="Title", index="MovieID", columns="Genres", aggfunc= np.size,).fillna(0)


def apply(x):
  if x >= 1:
    return 1
  else:
    return 0
    
m["Adventure"] = m["Adventure"].apply(apply)
m = m.astype(int)


# In[24]:


m


# In[25]:


final_data = df.merge(m,on="MovieID",how="left").drop(["Genres"],axis = 1)


# In[26]:


final_data


# In[27]:


final_data.MovieID = final_data.MovieID.astype(int)
final_data.UserID = final_data.UserID.astype(float)
final_data.Release_year = final_data.Release_year.astype(float)


# In[28]:


final_data.info()


# In[29]:


final_data.describe()


# In[30]:


final_data.describe(include="object")


# In[31]:


final_data.nunique()


# ---
# #### Unique values present in data
# ---
# - 6040 unique UserID
# - 7 different age groups
# - 21 occupations
# - 3439 different locations of users
# -3883 unique movies 
# 
# ---
# - There are movies available in database , which were never been watched by any user before . 
# - Thats is the reason we have lots of NaN values in our final dataset. 
# ---

# In[32]:


final_data.shape


# In[33]:


plt.rcParams["figure.figsize"] = (20,8)


# ## Most of the movies present in our dataset were released in year:
# 

# In[34]:


final_data.groupby("Release_year")["Title"].nunique().plot(kind="bar")
plt.show()


# In[35]:


# Number of Movies per Genres:

sns.barplot(m.sum(axis= 0).index,
            m.sum(axis= 0))


# In[36]:


m.sum(axis= 0) 


# In[37]:


final_data["Rating"].count()


# ## Number of movies Rated by each Gender type : 

# In[38]:


# Gender

asd = final_data.groupby("Gender")["Rating"].count() / final_data["Rating"].count() * 100
asd


# In[39]:


plt.pie(asd, labels = ["Female", "Male"],autopct='%1.1f%%')


# ## Users of which age group have watched and rated the most number of movies?

# In[40]:


plt.rcParams["figure.figsize"] = (10,6)
final_data.groupby("Age")["UserID"].nunique().plot(kind="bar")


# - in DataSet : majority of the viewers are  in age group of 25-34 
# - out of all , 25-34 age group have rated and watched the maximum number of movies. 
# - for other age groups data are as below: 

# In[41]:


final_data.groupby("Age")["MovieID"].nunique()


# In[42]:


plt.rcParams["figure.figsize"] = (10,8)
final_data.groupby("Age")["MovieID"].nunique().plot(kind="bar")


# ## Users belonging to which profession have watched and rated the most movies?
# 
# 

# In[43]:



plt.rcParams["figure.figsize"] = (20,8)

plt.subplot(121)
final_data.groupby("Occupation")["UserID"].nunique().sort_values().plot(kind="bar")
plt.subplot(122)
final_data.groupby("Occupation")["MovieID"].nunique().sort_values().plot(kind="bar")


# - Majority of the Users are College Graduates and Students , followed by Executives, educators and engineers. 
# y of the Users are College Graduates and Students , followed by Executives, educators and engineers. 
# - Maximum movies are watched and rated by user's occupations are College graduate students , writers , executives, educator and artists. 

# In[44]:


final_data.groupby("Occupation")["MovieID"].nunique().sort_values(ascending = False).head(6)


# In[45]:


final_data.columns


# ## Movie Recommendation based on Genres as per Majority Users occupation :     
# - below table shows the rank preference of each occupation users:
# - higher the number more prefered . 

# In[46]:


## Movie Recommendation based on Genre as per Majority Users : 


# In[47]:


np.argsort((final_data.groupby("Occupation")['Action', 'Adventure', 'Animation', "Children's", 
                                             'Comedy', 'Crime','Documentary', 'Drama', 'Fantasy', 
                                             'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Other',
                                             'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'].mean())   *100,axis = 1).loc[["writer","artist","academic/educator","executive/managerial","college/grad student"]]


# - Writers , artists and educator most preferes to watch Animation, Fantasy and Science Fiction movies, followed by Romance , Action and rest of the genres. 
# 
# - COllege Students most prefer to watch Children's , Science Fiction, Romance and Fantasy movies.
# 
# - Film-Noir is more prefered by the educators and Executive occupation users.

# ## what is the traffic on OTT, based on watch hour : 

# In[48]:


final_data.groupby("Watch_Hour")["UserID"].nunique().plot(kind="bar")


# ## Top 10 Movies have got the most number of ratings : 

# In[49]:


top10_movies = final_data.groupby("Title")["Rating"].count().reset_index().sort_values(by="Rating",ascending=False).head(10)


# In[50]:


top10_movies


# In[51]:


sns.barplot(y = top10_movies["Title"],
            x = top10_movies["Rating"])


# ## 5 Top rated Recommended Movies per each genre :

# In[52]:


Genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime','Documentary', 
          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Other','Romance', 
          'Sci-Fi', 'Thriller', 'War', 'Western']

for G in Genres:
  print(G)
  print("----------------------")
  print(final_data[final_data[G] == 1].groupby("Title")["Rating"].count().sort_values(ascending=False).head(5))
  print()
  print()
  print()


# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # Top 5 movie recommended as per age_Group based on ratings each age group provided 

# # 

# In[53]:


age_groups = final_data.Age.unique()


# In[54]:


for age_ in age_groups:
  print(age_)
  print("------")
  print(final_data[final_data.Age == age_].groupby("Title")["Rating"].count().sort_values(ascending=False).head())
  print()
  print()
  print()


# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# # 

# ## Creating a user Movie average rating Matrix : 

# In[92]:


df_.columns


# In[56]:


user_movie_rating_matrix = pd.pivot_table(df_,index = "UserID",
               columns = "Title",
               values = "Rating",
               aggfunc = "mean").fillna(0)
user_movie_rating_matrix.shape


# In[57]:


user_movie_rating_matrix


# ## item item similarity(hamming distance) based recommendation : 

# In[58]:


m = movies[["MovieID","Title","Genres"]]
m["Genres"] = m["Genres"].str.split("|")
m = m.explode("Genres")
m["Genres"] = m["Genres"].replace({"":"Other","Horro":"Horror","Sci-":"Sci-Fi","Sci":"Sci-Fi","Sci-F":"Sci-Fi","Dr":"Drama","Documenta":"Documentary",
                     "Wester":"Western","Fant":"Fantasy","Chil":"Children's","R":"Romance","D":"Drama","Rom":"Romance","Animati":"Animation","Childr":"Children's","Childre":"Children's",
                     "Fantas":"Fantasy","Come":"Comedy","Dram":"Drama","S":"Sci-Fi","Roma":"Romance","A":"Adventure","Children":"Children's","Adventu":"Adventure","Adv":"Adventure",
                      "Wa":"War","Thrille"  :"Thriller","Com":"Comedy","Comed":"Comedy","Acti":"Action","Advent":"Adventure","Adventur":"Adventure","Thri":"Thriller",
                        "Chi":"Children's","Ro":"Romance","F":"Fantasy","We":"Western","Documen":"Documentary","Music":"Musical","Children":"Children's" ,"Horr":"Horror",
                     "Children'":"Children's","Roman":"Romance","Docu":"Documentary","Th":"Thriller","Document":"Documentary"
                    })

m = m.pivot_table(values="Title", index="MovieID", columns="Genres", aggfunc= np.size,).fillna(0)


def apply(x):
  if x >= 1:
    return 1
  else:
    return 0
    
m["Adventure"] = m["Adventure"].apply(apply)
m = m.astype(int)


# In[59]:


m


# In[60]:



def Hamming_distance(x1,x2):
  return np.sum(abs(x1-x2))

Ranks = []
Query = "1"
for candidate in m.index:
  if candidate == Query:
    continue
  Ranks.append([Query,candidate,Hamming_distance(m.loc[Query],m.loc[candidate])])

Ranks = pd.DataFrame(Ranks,columns=["Query","Candidate","Hamming_distance"])
Ranks = Ranks.merge(movies[['MovieID', 'Title']], left_on='Query', right_on='MovieID').rename(columns={'Title': 'query_tittle'}).drop(columns=['MovieID'])
Ranks = Ranks.merge(movies[['MovieID', 'Title']], left_on='Candidate', right_on='MovieID').rename(columns={'Title': 'candidate_tittle'}).drop(columns=['MovieID'])
Ranks = Ranks.sort_values(by=['Query', 'Hamming_distance'])




Ranks.head(10)


# In[61]:



def Hamming_distance(x1,x2):
  return np.sum(abs(x1-x2))

Ranks = []
Query = "1485"
for candidate in m.index:
  if candidate == Query:
    continue
  Ranks.append([Query,candidate,Hamming_distance(m.loc[Query],m.loc[candidate])])

Ranks = pd.DataFrame(Ranks,columns=["Query","Candidate","Hamming_distance"])
Ranks = Ranks.merge(movies[['MovieID', 'Title']], left_on='Query', right_on='MovieID').rename(columns={'Title': 'query_tittle'}).drop(columns=['MovieID'])
Ranks = Ranks.merge(movies[['MovieID', 'Title']], left_on='Candidate', right_on='MovieID').rename(columns={'Title': 'candidate_tittle'}).drop(columns=['MovieID'])
Ranks = Ranks.sort_values(by=['Query', 'Hamming_distance'])




Ranks.head(10)


# In[62]:


movies = pd.read_fwf('C:/Users/USER/Downloads/zee-movies.dat', encoding='ISO-8859-1')
ratings =pd.read_fwf('C:/Users/USER/Downloads/zee-ratings.dat', encoding='ISO-8859-1')
users = pd.read_fwf('C:/Users/USER/Downloads/zee-users.dat', encoding='ISO-8859-1')

delimiter ="::"

users = users["UserID::Gender::Age::Occupation::Zip-code"].str.split(delimiter,expand = True)
users.columns = ["UserID","Gender","Age","Occupation","Zipcode"]

users["Age"].replace({"1": "Under 18","18": "18-24","25": "25-34",
                          "35": "35-44","45": "45-49","50": "50-55","56": "56+"},inplace=True)

users["Occupation"] = users["Occupation"].astype(int).replace({0: "other",1: "academic/educator",2: "artist",
                                                               3: "clerical/admin",4: "college/grad student",
                                             5: "customer service",6: "doctor/health care",7: "executive/managerial",
                                             8: "farmer" ,9: "homemaker",10: "K-12 student",11: "lawyer",
                                             12: "programmer",13: "retired",14: "sales/marketing",15: "scientist",
                                             16: "self-employed",17: "technician/engineer",
                                             18: "tradesman/craftsman",19: "unemployed",20: "writer"},
                                            )

delimiter ="::"

ratings = ratings["UserID::MovieID::Rating::Timestamp"].str.split(delimiter,expand = True)
ratings.columns = ["UserID","MovieID","Rating","Timestamp"]


movies.drop(["Unnamed: 1","Unnamed: 2"],axis = 1,inplace=True)

delimiter ="::"

movies = movies["Movie ID::Title::Genres"].str.split(delimiter,expand = True)
movies.columns = ["MovieID","Title","Genres"]

movies.shape,ratings.shape,users.shape

movies["Release_year"] = movies["Title"].str.extract('^(.+)\s\(([0-9]*)\)$',expand = True)[1]
movies["Title"] = movies["Title"].str.split("(").apply(lambda x:x[0])

from datetime import datetime
ratings["Watch_Hour"] =ratings["Timestamp"].apply(lambda x:datetime.fromtimestamp(int(x)).hour)
ratings.drop(["Timestamp"],axis = 1,inplace=True)

df = users.merge(movies.merge(ratings,on="MovieID",how="outer"),on="UserID",how="outer")
df["Genres"] = df["Genres"].str.split("|")
df = df.explode('Genres')

df["Genres"] = df["Genres"].replace({"":"Other","Horro":"Horror","Sci-":"Sci-Fi","Sci":"Sci-Fi","Sci-F":"Sci-Fi","Dr":"Drama","Documenta":"Documentary",
                     "Wester":"Western","Fant":"Fantasy","Chil":"Children's","R":"Romance","D":"Drama","Rom":"Romance","Animati":"Animation","Childr":"Children's","Childre":"Children's",
                     "Fantas":"Fantasy","Come":"Comedy","Dram":"Drama","S":"Sci-Fi","Roma":"Romance","A":"Adventure","Children":"Children's","Adventu":"Adventure","Adv":"Adventure",
                      "Wa":"War","Thrille"  :"Thriller","Com":"Comedy","Comed":"Comedy","Acti":"Action","Advent":"Adventure","Adventur":"Adventure","Thri":"Thriller",
                        "Chi":"Children's","Ro":"Romance","F":"Fantasy","We":"Western","Documen":"Documentary","Music":"Musical","Children":"Children's" ,"Horr":"Horror",
                     "Children'":"Children's","Roman":"Romance","Docu":"Documentary","Th":"Thriller","Document":"Documentary"
                    })
m = df.groupby(['MovieID','Genres'])['Title'].unique().str[0].unstack().reset_index().set_index('MovieID')
m = ~m.isna()
m = m.astype(int)


# ## Cosine Similarity : 

# ## Item and User :  -Cosine similarity Matrix : 

# In[63]:


from sklearn.metrics.pairwise import cosine_similarity 


# In[64]:


Item_similarity = cosine_similarity(user_movie_rating_matrix.T)
Item_similarity


# In[67]:


item_similarity.shape


# In[68]:


Item_similarty_matrix = pd.DataFrame(Item_similarity,
             index = user_movie_rating_matrix.columns,
             columns = user_movie_rating_matrix.columns)
Item_similarty_matrix 


# ## User Based Similartiy : 
# 

# In[69]:


User_similarity = cosine_similarity(user_movie_rating_matrix)
User_similarity.shape


# In[70]:


User_similarity


# In[71]:


User_similarity_matrix = pd.DataFrame(User_similarity,
             index = user_movie_rating_matrix.index,
             columns = user_movie_rating_matrix.index)
User_similarity_matrix 


# In[72]:


m


# ## Pearson Correlation
# 
# 

# In[73]:


correlated_movie_matrix = m.T.corr()


# In[74]:


correlated_movie_matrix


# In[75]:


movies[movies.MovieID == "1"]["Title"][0]


# In[76]:


movies[movies.Title.str.contains("Toy Story")].iloc[0].MovieID


# In[77]:


def recommend_movie_based_on_correlation(movie):
    TITLE = movies[movies.Title.str.contains(movie)].iloc[0]["Title"]
    
    INDEX = movies[movies.Title.str.contains(movie)].iloc[0].MovieID

    print(TITLE)
    print(INDEX)

    print(movies[movies.MovieID.isin(correlated_movie_matrix[INDEX].sort_values(ascending=False).head(10).index.to_list())]["Title"])


# In[78]:


recommend_movie_based_on_correlation("Toy Story")


# In[79]:


recommend_movie_based_on_correlation("Shawshank")


# In[80]:


recommend_movie_based_on_correlation("Titanic")


# In[81]:


recommend_movie_based_on_correlation("Braveheart")


# # k - Nearest Neighbours 

# In[82]:


from sklearn.neighbors import NearestNeighbors 


# In[83]:


kNN_model = NearestNeighbors(metric='cosine') 
kNN_model.fit(user_movie_rating_matrix.T)


# In[84]:


distances, indices = kNN_model.kneighbors(user_movie_rating_matrix.T, n_neighbors= 5)


# In[85]:


result = pd.DataFrame(indices)
result


# In[86]:


result.index = user_movie_rating_matrix.columns
result


# In[87]:


result.loc["Zero Effect "].to_list()


# In[88]:


movies.MovieID = movies.MovieID.astype("int32")


# In[89]:


movies[movies.MovieID.isin( result.loc["Zero Effect "].to_list())]


# ## Questions and Answers  :
# 
# 
# 1. Users of which age group have watched and rated the most number of movies?
# 
#     - age group 25-35
# 
# 
# 2. Users belonging to which profession have watched and rated the most movies?
#     - College Graduate Students and Other category
# 
# 3. Most of the users in our dataset who’ve rated the movies are Male. (T/F)
#     - Male
# 
# 4. Most of the movies present in our dataset were released in which decade?
#     - 90s
# 
# 
# 
# 5. The movie with maximum no. of ratings is ___.
#     - American Beauty	
# 
# 6. Name the top 3 movies similar to ‘Liar Liar’ on the item-based approach.
# 
#     - The Associate
#     - Ed's Next Move
#     - Bottle Rocket
#     - Mr. Wrong
#     - Cool Runnings
#     - Happy Gilmore
#     - That Thing You Do!
# 
# 
# 
# 
# 
# 7. On the basis of approach, Collaborative Filtering methods can be classified into ___-based and ___-based.
#       
#     - Memory based and Model based
#     
# 
# 
# 8. Pearson Correlation ranges between ___ to ___ whereas, Cosine Similarity belongs to the interval between ___ to ___.
#     - Pearson Correlation ranges between -1 to +1
#     - Cosine Similarity belongs to the interval between -1 to 1
# 
#     - similarity of 1 means that the vectors are identical, 
#     - a similarity of -1 means that the vectors are dissimilar,
#     - and a similarity of 0 means that the vectors are not similar.
# 
# 
# 9. Mention the RMSE and MAPE that you got while evaluating the Matrix Factorization model.
#     - Item-based Model : 
#     - RMSE: 0.8926
#     - User-based Model : 
#     - RMSE: 0.9345
# 
# 
# 
# 10. Give the sparse ‘row’ matrix representation for the following dense matrix -
# 
#     - [[1 0],[3 7]]
# 
#             ans  :
#                     [1 3 7]
#                     [0 0 1]
#                     [0 1 3]
#             

# In[90]:


from scipy.sparse import csr_matrix

dense_matrix = [[1,0],
                [3,7]]
sparse_matrix = csr_matrix(dense_matrix)


# In[91]:


print(sparse_matrix.data)
print(sparse_matrix.indices)
print(sparse_matrix.indptr)

