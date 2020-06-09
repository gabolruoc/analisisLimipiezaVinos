#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


# In[3]:


#Obtener datos desde csv
vinos = pd.read_csv('winequality-red.csv')
#Mostrar primeras filas de datos
vinos.head()


# In[4]:


#Número de filas y columnas
vinos.shape


# In[5]:


#Estadisticas desccriptivas
vinos.describe()


# In[6]:


#Número de valores duplicados
duplicados = vinos[vinos.duplicated()]
duplicados.shape


# In[7]:


#Tipo de datos
vinos.info()


# In[8]:


#Contar número de registros por quality
from collections import Counter
Counter(vinos['quality'])


# In[9]:


#Graficar el número de registros por quality
sns.countplot(x='quality', data=vinos)


# In[10]:


#Gráfico de correlaciones.
colormap = plt.cm.inferno
plt.figure(figsize=(12,12))
plt.title('Correlacion de atributos')
sns.heatmap(vinos.astype(float).corr(),
            linewidths=0.1,
            vmax=1.0,
            square=True, 
            linecolor='white',
            annot=True)


# In[11]:


#Muestra el numero de registros faltantes por columna
miss_values_count = vinos.isnull().sum(min_count=1)
miss_values_count = miss_values_count[miss_values_count != 0]

print(f"Número de columnas con datos faltantes: {miss_values_count.shape[0]}")
if miss_values_count.shape[0]:
    print("Recuento de valores nulos por columna: ")
    for name, miss_vals in miss_values_count.items():
        p = miss_vals > 1
        print(f"  - A la columna '{name}' le falta{'n' if p else ''} "
              f"{miss_vals} dato{'s' if p else ''}.")


# In[12]:


#Lista las columnas que componen el dataset
vinos.columns


# In[13]:


#Calcula el porcentaje de registros con datos 0
print(vinos[vinos == 0].count(axis=0)/len(vinos.index))


# In[14]:


#Grafica el boxplot de cada columna
plt.figure(figsize=(32,22))
plt.suptitle('Boxplots de cada atributo con outliers',fontsize=24)
for i in range(1,vinos.shape[1]+1):
    plt.subplot(3,4,i)
    plt.boxplot(vinos.iloc[:,i-1])
    plt.title(vinos.columns[i-1],fontsize=18)


# In[15]:


#Grafica el rango de valores outliers
plt.figure(figsize=(32,22))
plt.suptitle('Frontera de outliers usando 3 desviaciones estandar',fontsize=24)
for i in range(1,vinos.shape[1]+1):
    feature = vinos.iloc[:,i-1]
    mean = feature.mean()
    std_3 = feature.std()*3
    lower, upper = mean-std_3,mean+std_3
    plt.subplot(4,3,i)
    plt.hist(vinos.iloc[:,i-1],bins=50)
    plt.title(vinos.columns[i-1],fontsize=18)
    plt.axvspan(feature.min(),lower,color='red',alpha=0.3)
    plt.axvspan(upper,feature.max(),color='red',alpha=0.3)


# In[16]:


#Elimina los registros con valores outliers
outliers = vinos[np.abs(vinos[vinos.columns]-vinos[vinos.columns].mean())  <=  3 * vinos[vinos.columns].std()]
vinos_sin_outliers = outliers.dropna()
#Grafica histograma sin outliers
plt.figure(figsize=(32,22))
plt.suptitle('Outliers eliminados usando 3 desviaciones estandar',fontsize=24)
for i in range(1,vinos.shape[1]+1):
    feature = vinos.iloc[:,i-1]
    mean = feature.mean()
    std_3 = feature.std()*3
    lower, upper = mean-std_3,mean+std_3
    plt.subplot(4,3,i)
    plt.hist(vinos_sin_outliers.iloc[:,i-1],bins=50)
    plt.title(vinos.columns[i-1],fontsize=18)
    plt.axvspan(feature.min(),lower,color='red',alpha=0.3)
    plt.axvspan(upper,feature.max(),color='red',alpha=0.3)


# In[17]:


#Agrupa los registros en buen vino (1) y mal vino (0)
vinos_sin_outliers['categoria'] = pd.cut(vinos_sin_outliers['quality'], bins=[-np.inf,5, np.inf], labels=[0,1])


# In[18]:


#Grafica el conteo de registros por categoria
sns.countplot(x='categoria', data=vinos_sin_outliers)


# In[19]:


# Realiza el test de WIlcox de normalidad
pg.normality(vinos_sin_outliers)


# In[20]:


#Realiza el test de igualdad de varianzas para cada columna en base a la categoria
print('==============================')
print('fixed acidity')
print(pg.homoscedasticity(vinos_sin_outliers, dv='fixed acidity', group='categoria'))
print('==============================')
print('volatile acidity')
print (pg.homoscedasticity(vinos_sin_outliers, dv='volatile acidity', group='categoria'))
print('==============================')
print('citric acid')
print (pg.homoscedasticity(vinos_sin_outliers, dv='citric acid', group='categoria'))
print('==============================')
print('residual sugar')
print (pg.homoscedasticity(vinos_sin_outliers, dv='residual sugar', group='categoria'))
print('==============================')
print('chlorides')
print (pg.homoscedasticity(vinos_sin_outliers, dv='chlorides', group='categoria'))
print('==============================')
print('free sulfur dioxide')
print (pg.homoscedasticity(vinos_sin_outliers, dv='free sulfur dioxide', group='categoria'))
print('==============================')
print('total sulfur dioxide')
print (pg.homoscedasticity(vinos_sin_outliers, dv='total sulfur dioxide', group='categoria'))
print('==============================')
print('pH')
print (pg.homoscedasticity(vinos_sin_outliers, dv='pH', group='categoria'))
print('==============================')
print('sulphates')
print (pg.homoscedasticity(vinos_sin_outliers, dv='sulphates', group='categoria'))
print('==============================')
print('alcohol')
print (pg.homoscedasticity(vinos_sin_outliers, dv='alcohol', group='categoria'))
print('==============================')


# In[21]:


#Relaiza el test de Mann-Whitney U 
pg.mwu(vinos_sin_outliers[vinos_sin_outliers['categoria']== 0]['volatile acidity'],
       vinos_sin_outliers[vinos_sin_outliers['categoria']== 1]['volatile acidity'])


# In[22]:


#Relaiza el test de Mann-Whitney U 
pg.mwu(vinos_sin_outliers[vinos_sin_outliers['categoria']== 0]['residual sugar'],
       vinos_sin_outliers[vinos_sin_outliers['categoria']== 1]['residual sugar'])


# In[23]:


#Relaiza el test de Mann-Whitney U 
pg.mwu(vinos_sin_outliers[vinos_sin_outliers['categoria']== 0]['free sulfur dioxide'],
       vinos_sin_outliers[vinos_sin_outliers['categoria']== 1]['free sulfur dioxide'])


# In[24]:


#Relaiza el test de Mann-Whitney U 
pg.mwu(vinos_sin_outliers[vinos_sin_outliers['categoria']== 0]['pH'],
       vinos_sin_outliers[vinos_sin_outliers['categoria']== 1]['pH'])


# In[25]:


#Divide el data set en atributos(x) y variable resultante (y)
vinos_sin_outliers.pop('quality')
x = vinos_sin_outliers.iloc[:,:11]
y = vinos_sin_outliers['categoria']


# In[26]:


#Divide el dataset en dos grupos entrenamiento y test
from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


# In[27]:


#Realiza la regresion logistica
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)


# In[28]:


#Calcula matriz de confusion y la exactitud
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print(lr_conf_matrix)
print(lr_acc_score*100)


# In[29]:


#GRafica la matriz de confusión
plt.figure(figsize=(9,9))
sns.heatmap(lr_conf_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Categoria real');
plt.xlabel('Categoria clasificador');
all_sample_title = 'Exactitud: {0}'.format(lr_acc_score )
plt.title(all_sample_title, size = 15);


# In[30]:


#Realiza clasifica por árboles aleatorios
from sklearn.ensemble import RandomForestClassifier
RF_clf = RandomForestClassifier(n_estimators = 50)
cv_scores = cross_val_score(RF_clf,x_train, y_train, cv=10, scoring='accuracy')
RF_clf.fit(x_train, y_train)
pred_RF = RF_clf.predict(x_test)
cm = confusion_matrix(y_test, pred_RF)


# In[31]:


#Grafica la matriz de confusión
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Categoria real');
plt.xlabel('Categoria clasificador');
all_sample_title = 'Exactitud: {0}'.format(lr_acc_score )
plt.title(all_sample_title, size = 15);


# In[32]:


from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(x_train, y_train)
sel.get_support()
selected_feat= x_train.columns[(sel.get_support())]
len(selected_feat)
print(selected_feat)


# In[33]:


#Grafica la distribución de porcentaje de alcohol de acuerdo a la calidad del vino
g = sns.FacetGrid(vinos, col='quality')
g = g.map(sns.kdeplot, 'alcohol')


# In[34]:


#Grafica la distribución de cantidad de sulfatos de acuerdo a la calidad del vino
g = sns.FacetGrid(vinos, col='quality')
g = g.map(sns.kdeplot, 'volatile acidity')


# In[35]:


#Grafica la distribución de cantidad de acido cítrico de acuerdo a la calidad del vino
g = sns.FacetGrid(vinos, col='quality')
g = g.map(sns.kdeplot, 'residual sugar')


# In[36]:


g = sns.pairplot(vinos_sin_outliers, height=3,

                 vars=["volatile acidity", "residual sugar","alcohol"],hue = "categoria")


# In[37]:


vinos_sin_outliers.to_csv(r'red_wine_limpio.csv')


# In[ ]:




