#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
#from sklearn.model_selection import GridSearchCV
#from sklearn import ensemble
from sklearn.metrics import accuracy_score
import dabl
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


resultsDF = pd.read_csv('results.csv')
circuitsDF = pd.read_csv('circuits.csv')
driversDF = pd.read_csv('drivers.csv')
racesDF = pd.read_csv('races.csv')
constructorDF = pd.read_csv('constructors.csv')


# In[4]:


resultsDF.head()


# In[5]:


resultsDF.tail()


# In[6]:


circuitsDF.head()


# In[7]:


driversDF.head()


# In[8]:


racesDF.head()


# In[9]:


constructorDF.head()


# In[10]:


dfresul = pd.merge(resultsDF,driversDF,on='driverId')
dfresulcons = pd.merge(dfresul,racesDF,on='raceId')
dfresulrac = pd.merge(dfresulcons,constructorDF,on='constructorId')


# In[11]:


dfresulrac.head()


# In[12]:


dfresulcons.head()


# In[13]:


dfresul.head()


# In[14]:


dfresulrac.isnull().sum()


# In[15]:


dfresulrac = dfresulrac.drop(columns=['url_x','url_y','name_y','nationality_y','url','time_y'])
dfresulrac.head()


# In[16]:


dfresulrac.info()


# In[17]:


dfresulrac.describe()


# In[18]:


from IPython.display import Image
import os
Image('F1 race.PNG')


# In[19]:


import plotly.graph_objects as go
from plotly.offline import iplot


# In[20]:


#COUNTRIES WITH MORE WIN

fig = go.Figure(data=[go.Pie(labels=dfresulrac[(dfresulrac['position']== '1')].sort_values(by=['nationality_x'])['nationality_x'].unique(), 
                             values=dfresulrac[(dfresulrac['position']== '1')].groupby('nationality_x')['position'].value_counts(),hole=.3)])
fig.update_layout(title={
        'text': "Country with more wins",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, 
        template = "plotly_dark")
iplot(fig)


# In[21]:


#constructors with more wins
fig = go.Figure(data=[go.Pie(labels=dfresulrac[(dfresulrac['position']== '1')].sort_values(by=['constructorRef'])['constructorRef'].unique(), 
                             values=dfresulrac[(dfresulrac['position']== '1')].groupby('constructorRef')['position'].value_counts(),hole=.01)])
fig.update_layout(title={
        'text': "Counstructors with more wins",
        'y':0.9,
        'x':0.3,
        'xanchor': 'center',
        'yanchor': 'top'}, 
        template = "plotly_dark")
iplot(fig)


# In[22]:


#grand pix that hosted most number of races
fig = go.Figure(data=[go.Bar(
    x= racesDF['name'],
    y= racesDF['date'],
)])
fig.update_layout(title={
        'text': "grand pix that hosted most number of races",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, 
                  yaxis=dict(
                            title='date',
                            titlefont_size=16,
                            tickfont_size=14),
                  xaxis=dict(
                            title='grand pix',
                            titlefont_size=16,
                            tickfont_size=14),
                  template = "ggplot2"
                  )
iplot(fig)


# In[23]:


#constructor championship over the years
def championship_cons(year):
    f1_rep = dfresulrac[(dfresulrac['year']== year)].groupby('constructorRef')['points'].sum().reset_index()
    f1_rep = f1_rep.sort_values(by=['points'],ascending=False)
    fig = go.Figure(data=[go.Bar(
    x=f1_rep['constructorRef'],
    y=f1_rep['points']
    )])
    fig.update_layout(title={
        'text': f"Constructors' Championship ranking of {year}",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, 
                  yaxis=dict(
                            title='Points',
                            titlefont_size=16,
                            tickfont_size=14),
                  xaxis=dict(
                            title='Constructor',
                            titlefont_size=16,
                            tickfont_size=14),
                  template = "plotly_dark"
                  )
    return iplot(fig)


# In[24]:


championship_cons(2022)
championship_cons(2021)
championship_cons(2020)


# In[25]:


#drivers with more podiums
p1 = dfresulrac[(dfresulrac['position']== '1')].groupby('driverRef')['position'].value_counts()
p2 = dfresulrac[(dfresulrac['position']== '2')].groupby('driverRef')['position'].value_counts()
p3 = dfresulrac[(dfresulrac['position']== '3')].groupby('driverRef')['position'].value_counts()
#pilotos = dfresulrac.sort_values(by=['driverRef'])['driverRef'].unique()
driver1 = dfresulrac[(dfresulrac['position']== '1')].sort_values(by=['driverRef'])['driverRef'].unique()
driver2 = dfresulrac[(dfresulrac['position']== '2')].sort_values(by=['driverRef'])['driverRef'].unique()
driver3 = dfresulrac[(dfresulrac['position']== '3')].sort_values(by=['driverRef'])['driverRef'].unique()


# In[26]:


fig = go.Figure(go.Bar(x=driver1, y=p1, name='1º place'))
fig.add_trace(go.Bar(x=driver2, y=p2, name='2º place'))
fig.add_trace(go.Bar(x=driver3, y=p3, name='3º place'))

fig.update_layout(barmode='stack',title={
        'text': "driver with more podiums",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, 
                  yaxis=dict(
                            title='Podiums',
                            titlefont_size=16,
                            tickfont_size=14),
                  xaxis=dict(
                            title='Driver',
                            titlefont_size=16,
                            tickfont_size=14),
                  template = "plotly_dark"
                  )
iplot(fig)


# In[27]:


nationality_counts = driversDF['nationality'].value_counts().to_frame().reset_index().rename(columns={'index':'nationality', 'nationality':'count'})
nationality_counts.head(20)


# In[28]:


#Top 20 driver nationalities in F1
import plotly.express as px
fig = px.pie(nationality_counts.head(20), values='count', names='nationality', 
             title='Top 20 driver nationalities in F1', hole = 0.2,
            color_discrete_sequence=px.colors.qualitative.Pastel)

fig.show()


# In[29]:


#F1 circuits across the globe
get_ipython().system('pip install folium')
import pandas as pd
import folium
coordinates=[]
for lat,lng in zip(circuitsDF['lat'],circuitsDF['lng']):
    coordinates.append([lat,lng])
maps = folium.Map(zoom_start=3,tiles='Stamen Terrain')  #map_types (Stamen Terrain, Stamen Toner, Mapbox Bright, cartodbpositron)
for i,j in zip(coordinates,circuitsDF.name):
    marker = folium.Marker(
        location=i,
        icon=folium.Icon(icon="star",color='purple'),
        popup="<strong>{0}</strong>".format(j))  #strong is used to bold the font (optional)
    marker.add_to(maps)
maps


# In[30]:


sum_driver = dfresulrac.groupby(['year','driverRef'])['points'].sum().reset_index()
champions = sum_driver.loc[sum_driver.reset_index().groupby(['year'])['points'].idxmax()]
champions = champions['driverRef'].value_counts().reset_index()
champions.rename(columns={'index':'driver','driverRef':'titles'}, inplace = True)


# In[31]:


fig = go.Figure(data=[go.Bar(
    x=champions['driver'],
    y=champions['titles']
    )])
fig.update_layout(title={
        'text': "Drivers with more titles",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, 
                  yaxis=dict(
                            title='Titles',
                            titlefont_size=16,
                            tickfont_size=14),
                  xaxis=dict(
                            title='Driver',
                            titlefont_size=16,
                            tickfont_size=14),
                  template = "plotly_dark"
                  )
iplot(fig)


# In[ ]:




