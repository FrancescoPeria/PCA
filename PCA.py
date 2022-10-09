# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 11:34:34 2022

@author: peria

Topic: Principal Component Analysis

About this file --> The datasets includes:
- Participants: participant number
- Four columns for the second-order strengths factors previously extracted via PCA:
    - Openness
    - Restraint
    - Transcendence
    - Interpersonal
- The three dependent measures: DASS21 (Depression Anxiety and Stress Scale) 
- GHQ12 (General Health Questionnaire)
- SEC (Self-efficacy for Covid-19)
- Six demographic variables added in the analysis:
- Age
- Gender
- Work (representing the perceived work change subsequent to lockdown)
- Student (being a student or not)
- Day (how many days passed when the participant responded since the day the survey was opened)

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
#sns.set_style('whitegrid')

#%% Importing df

df = pd.read_excel(r'C:\Users\peria\Desktop\DATA SCIENCE\MODELLI\PCA_Covid_Psyco_Survey.xlsx', index_col = 0)
print('\n', df)
print('\n', df.columns)

# Preprocessing of features
df.drop(columns = ['Day', 'DASS_21'], inplace = True)
df.dropna(inplace = True)
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
df['Student'] = df['Student'].map({'Other': 0, 'Student': 1})

# Understanding variable tipology
print(df.info())

#%% Plotting correlation of variables. Ci sono correlazioni interessanti

fig = plt.figure(figsize=(15, 12))
plt.matshow(df.corr(), fignum = fig.number, cmap='seismic')
plt.xticks(range(df.shape[1]), df.columns, fontsize = 12, rotation = 90)
plt.yticks(range(df.shape[1]), df.columns, fontsize = 12)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
plt.title('Matrice di correlazione', fontsize = 17)

#%% Scalo le features per farci la PCA
import sklearn
from sklearn.preprocessing import StandardScaler

cols = df.columns

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
df = pd.DataFrame(scaled_data, columns = cols)

print('\nAdesso il df è standardizzato:\n', df.head())


#%% PCA

from sklearn.decomposition import PCA

PCA = PCA()
PCA.fit(scaled_data)
PCA_data = PCA.transform(scaled_data)


#%% Scree Plot to understand how much PCs will go in the final graph

# percentage of explained variance (lambda_1 + lambda_k) / (lambda_1 + ... + lambda_d)
perc_var = PCA.explained_variance_ratio_
print(perc_var)

labels = ['PC' + str(j) for j in range(1, len(perc_var) + 1)]

plt.bar(x = labels, height = perc_var, color = 'dodgerblue')
plt.xticks(rotation = 90, fontsize = 8)
plt.xlabel('Principal Components')
plt.ylabel('% of explained variance')
plt.title('Scree Plot')
plt.show()

#%% Creo il df con le osservazioni viste nel sistema di riferimento delle CPs
# Le colonne sono le CP
# Le righe sono le osservazioni raccolte nel df originario

df_PCA = pd.DataFrame(PCA_data, columns = labels, index = df.index)
print('\nEcco il df_PCA che contiene le coordinate delle osservazioni rispetto alle CPs:\n', df_PCA)

plt.scatter(df_PCA['PC1'], df_PCA['PC2'], s = 5, c = 'dodgerblue')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Observations projected on principal space PC1 - PC2')
plt.show()


#%% LOADING SCORES per vedere il legame tra i fattori originari e le PCs

# Legame tra fattori originari e PC1:
    
# PCA.components_ mi dà le coordinate delle direzioni principali, ovvero gli autovettori della 
# matrice di covarianza dei dati df.values tali che lungo di esse ho la maggiore variabilità


# Legame tra fattori originari e PC1:
print('\nLink between PC1 and original factors:')
principal_direction_1 = pd.Series(PCA.components_[0], index = cols)
idx_sorted = principal_direction_1.abs().sort_values(ascending = False).index
principal_direction_1 = principal_direction_1[idx_sorted]

# Legame tra fattori originari e PC2:
print('\nLink between PC2 and original factors:')
principal_direction_2 = pd.Series(PCA.components_[1], index = cols)
idx_sorted = principal_direction_2.abs().sort_values(ascending = False).index
principal_direction_2 = principal_direction_2[idx_sorted]


#%% Significative plot

fig, ax = plt.subplots(1, 2, figsize = [20,6])

factor_1 = 'Transcendence'
factor_2 = 'Openness'

ax[0].scatter(df[factor_1], df[factor_2], s = 5, c = 'dodgerblue')
ax[0].set_xlabel(factor_1)
ax[0].set_ylabel(factor_2)


# Considering PC1, PC2, X1, X2, V1, V2 be column vectors --> [PC1 PC2] = [X1 X2][V1 V2]
# PC1 = X1_1 * V1_1 + X2_1*V1_2

for j in range(df.shape[0]):
    ax[0].scatter( df_PCA['PC1'][j] * principal_direction_1[factor_1],
                   df_PCA['PC1'][j] * principal_direction_1[factor_2],
                   s = 5, c = 'red')
    
for j in range(df.shape[0]):
    ax[0].scatter( df_PCA['PC2'][j] * principal_direction_2[factor_1],
                   df_PCA['PC2'][j] * principal_direction_2[factor_2],
                   s = 5, c = 'green')



ax[1].scatter(df_PCA['PC1'], df_PCA['PC2'], s = 5, c = 'dodgerblue')
ax[1].set_xlabel('PC1')
ax[1].set_ylabel('PC2')

plt.show()

