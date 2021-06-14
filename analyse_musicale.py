
# coding: utf-8

# # LOKO LOÏC 15607684 Framework Logiciel pour Big Data (Semestre 1)

# # Importation des librairies
# 
# Nous commençons premièrement à importer les différenters librairies nécessaires pour le travail, notamment pandas, numpy, matplot etc.

# In[106]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
from scipy.stats import pearsonr
get_ipython().magic('matplotlib inline')


# # Lecture du fichier
# 
# Ensuite nous lisons nos données par le biais de la fonction pd.read_csv, ensuite nous jettons un oeil dans nos données

# In[5]:


df=pd.read_csv('top2018.csv')


# In[8]:


df.head()


# # Traitement des données
# 
# Nous allons maintenant voir ce qui nos ces top sons si populaires.
# Ci-dessous une carte de chaleur qui montre les corrélatiions entre certaines colonnes, et cela grâce à la fonction de pandas corr().

# In[9]:


df['Duration_min']=df['duration_ms']/60000


# In[ ]:


df.drop(columns='duration_ms',inplace=True)


# In[53]:


Correlation= df.drop(['id','name','artists'], axis=1)
plt.figure(figsize = (15,6))
sns.heatmap(Correlation.corr(),annot=True,cmap="PuBuGn")


# Nous pouvons observer dans les tons les plus forts les corrélations existantes entre les différentes colonnes, à première vue les colonnes loudness et énergy, ensuite entre valence et danceability, nous allons analyser les forces de ces variables.
# Remémorons nous le top 15 des artistes.

# ### top 15 des artistes

# In[94]:


df['artists'].value_counts().head(15)


# 
# ## Analyse de la dansabilité

# In[26]:


sns.set_style(style='whitegrid')
sns.distplot(df['danceability'])


# Nous avons donc une dansabilité moyenne de: 0.7164600000000001.
# 
# On peut donc constater que la majorité des chansons ont une dansabilité assez elevée, on peut donc en déduire que les utilisateurs adorent les musiques qui leur permet de danser car plus le taux est elevé plus la musique est dansante. Nous pouvons donc supposer ces titres supérieurs à 0.50 peut être beaucoup streamer en soirées, fêtes etc.
# 
# On peut décider de faire un classification selon laquelle on aura les titres sécouants, dansants, et peu dansants : 

# In[33]:


secouant=df['danceability'] >= 0.71
dansant=(df['danceability'] >= 0.5) & (df['danceability'] < 0.71)
calme=df['danceability'] < 0.5

data=[secouant.sum(),dansant.sum(),calme.sum()]
pd.DataFrame(data,columns=['Taux'],index=['Sécouant','Dansant','Calme'])


# ## Analyse de l'énergie

# In[36]:


sns.set_style(style='darkgrid')
sns.distplot(df['energy'])


# Nous avons donc une énergie moyenne de: 0.6590600000000001.
# 
# On peut donc constater que la majorité des chansons ont une énérgie elevée, on peut donc en déduire que les utilisateurs adorent les musiques qui leur permet de danser car plus le taux est elevé plus la musique est énergétique. Nous pouvons donc supposer que le public est assez énergetique. On donc a une idée sur sa tranche d'âge

# ## Analyse de la durée

# In[37]:


sns.distplot(df['duration_ms'])


# Nous avons donc une durée moyenne de: 205206.78.
# 
# La valeur moyenne de la durée est de 3 minutes et 25 secondes.
# Les gens n'aiment donc pas les chansons trop courtes ou trop longues. 

# # Le Tempo
# 
# Ci-dessous les 5 caractéristique d'un Tempo
#         
#         -Allant : au rythme, calme, un peu vif 76 - 108 bpm
#         -Lenteur : très lent 20 bpm
#         -Allegro : animé et rapide. 110 - 168 bpm
#         -Adagio : lent et majestueux 66 - 76 bpm
#         -Moderato: Modéré 88-112 bpm
#         -Rapide : très rapide 168 - 200 bpm
# 

# In[103]:


df['Rhythm']=df['tempo']

df.loc[(df['tempo']>=76) & (df['tempo']<=108),'Rhythm']='Allant'
df.loc[df['tempo']<50,'Rhythm']='lent'
df.loc[(df['tempo']>=110) & (df['tempo']<=168),'Rhythm']='Allegro'
df.loc[(df['tempo']>=66) & (df['tempo']<=76),'Rhythm']='Adagio'
#df.loc[(df['tempo']>=88) & (df['tempo']<=112),'Rhythm']='Moderato'
df.loc[df['tempo']>168,'Rhythm']='Rapide'


# In[104]:


df['Rhythm'].value_counts()


# In[105]:


sns.set_style(style='whitegrid')
Rhy=df['Rhythm'].value_counts()
Rhy_DF=pd.DataFrame(Rhy)
sns.barplot(x=Rhy_DF.Rhythm, y=Rhy_DF.index, palette="viridis")
plt.title('Tempo')


# # Correspondance
# 
# Ici nous utilisons les variables les plus importantes pour voir les correspondances.

# In[74]:


Correspondance=df[['danceability','energy','valence','loudness','tempo']]


# In[75]:


sns.heatmap(Correspondance.corr(),annot=True,cmap="PuBuGn")


# In[76]:


sns.jointplot(data=Correspondance,y='loudness',x='energy',kind='reg',stat_func=pearsonr)


# Nous pouvons  observer que comme les valeurs de l'intensité sonore sont proches de zéro.
# L'énergie et l'intensité sonore semblent bien corrélées,toute fois l'énergie et la dansabilité correspondent peu, étonnant.

# # Les Top selons les caractéristiques

# # # Le top 10 des titres avec des vibes positives

# In[77]:


df['Rhythm']=df['tempo']


# In[78]:


df[['name','artists','energy','valence','tempo','Rhythm']].sort_values(by='valence',ascending=False).head(10)


# ## Le top 10 des titres les plus dansants

# In[79]:


df[['name','artists','danceability','valence','tempo','Rhythm']].sort_values(by='danceability',ascending=False).head(10)


# # Caractéristiques du meilleur artiste

# In[80]:



df['artists'].value_counts().head(4)


# In[81]:


XXXTENT=df[df['artists']=='XXXTENTACION']
XXXTENT[['name','danceability','energy','loudness','valence','tempo','Rhythm']]


# # Conclusion
# 
# On peut remarquer la plupart des autres chansons ont beaucoup de similarités, par conséquent, la plupart des auditeurs et ceux qui écoutent les chansons en streaming préfèrent ces goûts musicaux similaires, sauf biensure celles qui n'y sont pas car spéciales ( notamment Yes indeed de Lil Baby, changes de XXXTentaction ou, Lovely de Bilie Elish).
# Une forte raison aussi est l'époque de la chanson, la majorité des chansons étaient dans les formes de tempo Allegro et Allant qui caractérise le Hip Hop, la Pop, le Reggae et le rap.
