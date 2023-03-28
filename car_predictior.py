import pandas as pd
import datetime
import pickle
import streamlit as st
from Zee Case Study import *
import yfinance as yf
movies = pd.read_fwf('C:/Users/USER/Downloads/zee-movies.dat',
                     encoding='ISO-8859-1')
ratings =pd.read_fwf('C:/Users/USER/Downloads/zee-ratings.dat',
                     encoding='ISO-8859-1')
users = pd.read_fwf('C:/Users/USER/Downloads/zee-users.dat',
                    encoding='ISO-8859-1')
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

st.dataframe(movies)
movie = st.selectbox('Select the movie', movies["Title"].to_list())
if (st.button('Recommended Movies')):
   # execfile("D:/Scaler/Projects/Zee Case Study.ipynb")
   # with open("D:/Scaler/Projects/Zee Case Study.ipynb", 'r') as f:
    #   exec(f.read())
    #exec(open("D:/Scaler/Projects/Zee Case Study.ipynb").read())
    lis = recommend_movie_based_on_correlation(movie)
    st.text(lis)
