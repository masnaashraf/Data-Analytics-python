#!/usr/bin/env python
# coding: utf-8

# # Data Science Regression Project:Bangalore House Price Prediction

# # Importing  Liabraries:

# In[90]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


# # Data Importing/Data Load :

# Loading Banglore home prices into a data frame df.

# In[91]:


df=pd.read_csv("C://Users//Masna_2//Desktop//data Analytics//projects//Bengaluru_House_Data.csv")


# # Studying the DataFrame

# viewing first 5 rows of the data set

# In[3]:


df.head()


# checking number of rows and columns of the data set

# In[4]:


df.shape


# Viewing names of columns presen in the data set

# In[5]:


df.columns


# In[ ]:


#getting the information about the dataset


# In[13]:


df.info()


# finding the statistical information about the data

# In[6]:


df.describe()


# checking the null values

# In[7]:


df.isnull().sum()


# In[ ]:


Drop features that are not required to build our model


# In[92]:


df1=df.drop(["area_type","society","balcony","availability"],axis=1)


# In[93]:


df1


# # Data Cleaning: Handle NA values

# In[9]:


df1.isnull().sum()


# In[ ]:


#droping row where size and bath is naan 


# In[94]:


df1=df1.dropna()


# In[95]:


df1.isnull().sum()


# # Explratory Data Analysis

# In[96]:


df1.info()


# In[ ]:


viewing unique values of each column


# In[97]:


for column in df1.columns[:]:
    print(column,": ",df1[column].unique())


# # Feature Engineering

# Add new Integer Feature,bhk for the column size

# In[98]:


df1['bhk']=df1['size'].apply(lambda x: int(x.split(" ")[0]))
df1.bhk.unique()


# In[99]:


df1.head()


# dealing the total_sqft column

# In[17]:


df1.total_sqft.unique()


# In[39]:


#presence of range value like this '1133 - 1384' .so need to convert it into single value


# In[40]:


#function to check value is float or not


# In[100]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
    


# In[101]:


#negate operation is used to return rows  where total_sqft value is not float


# In[102]:


df1[~df1["total_sqft"].apply(is_float)].head(10)


# In[44]:


#converting the range value to average


# In[103]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return(float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[21]:


convert_sqft_to_num('2100-2850')


# In[47]:


convert_sqft_to_num('2100')


# In[104]:


df2=df1.copy()


# In[105]:


df2["total_sqft"]=df2["total_sqft"].apply(convert_sqft_to_num)


# In[106]:


df2.head()


# In[25]:


df2.loc[30]


# In[ ]:


#adding new feature called price per square feet


# In[107]:


df3=df2.copy()


# In[108]:


df3["price_per_sqft"]=df3["price"]*100000/df3['total_sqft']
df3.head()


# In[28]:


plt.figure(figsize=(20,10))
sns.barplot(x='bhk', y = 'price_per_sqft',data = df3)


# # Dimensionality Reduction

# In[109]:


len(df3.location.unique())


# In[ ]:


#We will use Dimensionality Reduction to reduce the numbeer of Locations

#Here dimensionality is a categorical variable


# In[58]:


#Stripping extra space(leading and trailing)using lambds function


# In[110]:


df3.location=df3.location.apply(lambda x:x.strip())
location_stats=df3['location'].value_counts().sort_values(ascending=False)
location_stats


# In[30]:


#now we are checking location < 10 data points


# In[111]:


len(location_stats[location_stats<=10])


# In[62]:


#considering these these locations as other location


# In[63]:


#To reduce the number of locations, we can say that any location that has less than 10 data points is called other location


# In[112]:


loc_stat_lessthan10=location_stats[location_stats<=10]
loc_stat_lessthan10


# In[113]:


df3.location=df3.location.apply(lambda x:"other" if x in loc_stat_lessthan10 else x)
len(df3.location.unique())


# In[34]:


df3.head(10)


# In[35]:


df3.shape


# # Outlier Detection and Removal:
# 
# 

# #Outlier Detection:
# #Outliers are not errors but really large or small values which make no sense in the data. 
# 

# In[ ]:


#considering a bedroom ususally take 300sqft


# In[ ]:


#outlier removal:


# #keeping only datapoints where square feet of bedroom doesnot exceed the total squre feet by considering sqft of a bedroom
# as 300sqft

# 
# #If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier.
# #We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft

# In[114]:


df3[df3.total_sqft/df3.bhk<300].head()


# In[37]:


plt.figure(figsize=(20,10))
sns.barplot(x='bhk', y ='total_sqft',data = df3)


# In[ ]:


#Check above data points. We have 6 bhk apartment with 1020 sqft. Another one is 8 bhk and total sqft is 600. These are clear data errors that can be removed safely


# In[115]:


df4=df3[~(df3.total_sqft/df3.bhk<300)]


# In[116]:


df4.shape


# In[40]:


df4.head()


# In[83]:


plt.figure(figsize=(20,10))
sns.barplot(x='bhk', y ='total_sqft',data = df4)


# In[ ]:


#checking column price per squarefeet


# In[41]:


df4.price_per_sqft.describe()


# In[ ]:


#Outlier Removal Using Standard Deviation and Mean


# In[ ]:


#clearly the minimum value of square feet cannot be 267 rupees and maximum cannot be 176470


# In[ ]:


#Basically what the below function does is take the data points per location and filter out the data points that have standard deviation that is greater than 1


# In[ ]:


#keeping only datapoints on one standard deviation (68%)


# In[117]:


def remove_outlier_pricepersqft(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        meanx=np.mean(subdf.price_per_sqft)
        stdx=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(meanx-stdx)) & (subdf.price_per_sqft<=(meanx+stdx))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out


# In[118]:


df5=remove_outlier_pricepersqft(df4)
df5.shape


# In[ ]:


#One more thing that we have to check is that if the price of a two bhk apt is greater than 3bhk apt for the same square foot area

#We are going to plot a scatter plot which will tell us how many of these types of points we have


# In[258]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50) # s is the marker size
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df5,"Rajaji Nagar")


# In[ ]:


#for around 1700 sq foot area 4 of the two bedroom apt price is higher than 3 bedroom


# In[144]:


plot_scatter_chart(df5,"Hebbal")


# We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). What we will do is for a given location, we will build a dictionary of stats per bhk, i.e.
# 
# {
# '1' : {
# 'mean': 4000,
# 'std: 2000,
# 'count': 34
# },
# '2' : {
# 'mean': 4300,
# 'std: 2300,
# 'count': 22
# },
# }
# Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment

# In[119]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df6 = remove_bhk_outliers(df5)
df6.shape


# In[79]:



plot_scatter_chart(df6,"Rajaji Nagar")


# In[146]:


plot_scatter_chart(df6,"Hebbal")


# In[ ]:


#trying to find out number of appartment per sqaure foot area


# In[147]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df6.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[ ]:


#so from above histogram we can see that majority of datapoints lies between rs 0 to rs 10000 price per square feet 


# In[ ]:


#now checking for dathroom column


# In[120]:


df6.bath.unique()


# In[162]:


plt.hist(df6.bath,rwidth = 0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[121]:


df6[df6.bath>10]


# In[ ]:


#trying to remove records  if no of bath>no of bedroom + 2 


# In[122]:


df6[df6.bath>df6.bhk+2]


# In[ ]:


#we can see 4 bedroom with 8 bathroom,3 bedroom with 6 bathroom .So we can consider these as outliers


# In[123]:


df7 = df6[df6.bath<df6.bhk+2]
df6.shape


# In[ ]:


#considering the balcony column


# In[48]:


df7.head()


# In[124]:


df7.shape


# In[ ]:


#dropping unnecessary columns


# In[125]:


df8=df7.drop(["size","price_per_sqft"],axis=1)


# In[126]:


df8.shape


# size and price_per_sqft can be dropped because they were used only for outlier detection. Now the dataset is neat and clean and we can go for machine learning training

# In[203]:


df8.head(20)


# # Encoding  of chategorical value and Machine Learning Model

# In[127]:


dummies = pd.get_dummies(df8,['location'],drop_first=True)
dummies.head()


# In[128]:


df=dummies
df.head()


# # applying standard scalar to standardize data

# In[53]:


from sklearn.preprocessing import StandardScaler


# In[54]:


# define standard scaler
scaler = StandardScaler()
  
# transform data
dfscaled = scaler.fit_transform(df)


# In[55]:


dfnew=pd.DataFrame(dfscaled,columns=df.columns)
dfnew.head()


# In[ ]:


#sepertating dependend and independendvariables


# In[56]:


dfnew.shape


# In[57]:


Y=dfnew.price
Y.head()


# In[129]:


Y=df.price
Y.head()


# In[130]:


X = dfnew.drop(['price'],axis=1)
X.head(3)


# In[131]:


X = df.drop(['price'],axis=1)
X.head(3)


# In[59]:


X=df.drop(['price'],axis=1)
X.head()


# # Model selection and applying machine learning algorim

# In[132]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)


# Applying Linear Regresion

# In[133]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf_score=lr_clf.score(X_test,y_test)
lr_clf_score


# In[151]:


lr_clf.predict([['location_1st Phase JP Nagar',1000,2,2]])


# # finding cross validation score

# In[134]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(lr_clf,X_train,y_train,cv=5)
score


# In[ ]:


Applying decision tree  regressor algorithm


# In[135]:


from sklearn.tree import DecisionTreeRegressor
dec_reg=DecisionTreeRegressor(random_state=0)
dec_reg.fit(X_train,y_train)
dec_reg_score=dec_reg.score(X_test,y_test)
dec_reg_score


# Applying KNeighbors Regressor

# In[136]:


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
knn_score=knn.score(X_test,y_test)
knn_score


# Applying random forest regressor

# In[137]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=100)
rfr.fit(X_train, y_train)
rfr_score=rfr.score(X_test,y_test)
rfr_score


# # Comparing Accuracy score with different algorithm

# In[138]:


Model_Comparison = pd.DataFrame({
    'Model' : ['Linear Regression','Random Forest Regression','Decision Tree Regression','KNeighbors regression'],
    'Scores_test': [lr_clf_score* 100 ,rfr_score* 100,dec_reg_score* 100,knn_score* 100]
    })
Model_Comparison


# Based on above results we can say that LinearRegression gives the best score. Hence we will use that.

# # Testing the model

# In[139]:


print(X.columns)


# In[140]:


my_list=X.columns.values.tolist()
print(my_list)


# In[141]:


'location_1st Block Jayanagar' in my_list


# In[142]:


np.where(X.columns=='location_Yeshwanthpur')[0][0]


# In[143]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[145]:


predict_price('location_1st Phase JP Nagar',1000,2,2)


# In[146]:


predict_price('location_1st Phase JP Nagar',1500,5,5)


# In[146]:


predict_price('location_1st Phase JP Nagar',1500,5,5)

