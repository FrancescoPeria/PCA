# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 23:48:07 2022

@author: peria

About this file --> The datasets includes:
- Participants: participant number
- Four columns for the second-order strengths factors previously extracted via PCA:
    - Openness
    - Restraint
    - Transcendence
    - Interpersonal
- The three dependent measures: 
    - DASS21 (Depression Anxiety and Stress Scale) 
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
sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
#sns.set_style('whitegrid')

#%% Importing df


df = pd.read_excel(r'C:\Users\peria\Desktop\DATA SCIENCE\MODELLI\PCA_Covid_Psyco_Survey.xlsx', index_col = 0)
print('\n', df)
print('\n', df.columns)

# Preprocessing of features
df.drop(columns = ['Day', 'DASS_21', 'Openness', 'Restraint',
                   'Transcendence', 'Interpersonal'], inplace = True)

df.dropna(inplace = True)
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
df['Student'] = df['Student'].map({'Other': 0, 'Student': 1})

# Understanding variable tipology
print(df.info())

print('\n', '*'*40, '\n')



#%% Plotting correlation of variables

print('\n\nPlotting correlation of variables\n')

fig = plt.figure(figsize=(15, 12))
plt.matshow(df.corr(), fignum = fig.number, cmap='seismic')
plt.xticks(range(df.shape[1]), df.columns, fontsize = 12, rotation = 90)
plt.yticks(range(df.shape[1]), df.columns, fontsize = 12)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
plt.title('Correlation Matrix', fontsize = 17)
plt.show()

print('\n', '*'*40, '\n')

#%% Scalo le features per farci la PCA

import sklearn
from sklearn.preprocessing import StandardScaler

cols = df.columns

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
df = pd.DataFrame(scaled_data, columns = cols)

print('\nAdesso il df è standardizzato:\n', df.head())

print('\n', '*'*40, '\n')


#%% Calculate the PCs using the class PCA from scratch

from PCA_from_scratch import PCA_from_scratch

k = 10

# Create object
PCA_from_scratch = PCA_from_scratch(k_components = k)

# Fit method
PCA_from_scratch.fit(df.values)

# Transform method
PCs = PCA_from_scratch.transform(df.values)

df_PCA_from_scratch = pd.DataFrame(np.real(PCs), 
                                   columns = ['PC' + str(j) for j in range(1, k+1)])

print('\n\nPrincipal Components PCs:\n', df_PCA_from_scratch)


print('\n', '*'*40, '\n')


#%% Find the first 2 principal directions which are commonly known as LOADINGS.
# By to obtain loadings I should multiply the principal direction (i.e. the eigenvector of covariance matrix)
# by eigenvalue^0.5. In every case also from the principal directions I can understand how each original factor
# enters into the principal directions itself

principal_direction_1 = pd.Series(np.real(PCA_from_scratch.principal_directions[0]), index = cols)
idx_sorted = principal_direction_1.abs().sort_values(ascending = False).index
principal_direction_1 = principal_direction_1[idx_sorted]


principal_direction_2 = pd.Series(np.real(PCA_from_scratch.principal_directions[1]), index = cols)
idx_sorted = principal_direction_2.abs().sort_values(ascending = False).index
principal_direction_2 = principal_direction_2[idx_sorted]


principal_direction_3 = pd.Series(np.real(PCA_from_scratch.principal_directions[2]), index = cols)
idx_sorted = principal_direction_3.abs().sort_values(ascending = False).index
principal_direction_3 = principal_direction_3[idx_sorted]


principal_direction_4 = pd.Series(np.real(PCA_from_scratch.principal_directions[3]), index = cols)
idx_sorted = principal_direction_4.abs().sort_values(ascending = False).index
principal_direction_4 = principal_direction_4[idx_sorted]


# Legame tra fattori originari e PC2:
print('\n\nLink (sorted basing on importance) between PC1 and original factors:\n', principal_direction_1)
print('\n\nLink (sorted basing on importance) between PC2 and original factors:\n', principal_direction_2)
print('\n\nLink (sorted basing on importance) between PC3 and original factors:\n', principal_direction_3)
print('\n\nLink (sorted basing on importance) between PC4 and original factors:\n', principal_direction_4)

print('\n', '*'*40, '\n')


#%% Scree Plot to understand how much PCs will go in the final graph

print('\n\nScree Plot to understand how much PCs will go in the final graph\n')

fig, ax = plt.subplots()

axt = ax.twinx()

# percentage of explained variance (lambda_k) / (lambda_1 + ... + lambda_d)
perc_var = np.real(PCA_from_scratch.explained_variance_ratio)
print(perc_var)

labels = ['PC' + str(j) for j in range(1, len(perc_var) + 1)]
ax.bar(labels, perc_var, color = 'dodgerblue')


# cumulative percentage of explained variance (lambda_1 + lambda_k) / (lambda_1 + ... + lambda_d)
cum_var = np.cumsum(perc_var) / np.sum(perc_var)
axt.plot(labels, cum_var, color = 'blue', marker = '*')

ax.set_xlabel('Principal Components')
ax.set_ylabel('% of explained variance')
ax.set_title('Scree Plot')

axt.set_ylim([0, 1.1])
axt.grid(False)
plt.show()


print('\n', '*'*40, '\n')


#%% PROOF that the eigenvalues are the variance of data on the first principal directions

list_variance_along_prin_dir = []

# Factors are on te columns while on the rows I see observations
covariance_matrix = np.cov(df.values, rowvar = False)

for j in range(k):
    variance_along_prin_dir = PCA_from_scratch.principal_directions[j].T \
                              @ covariance_matrix \
                              @ PCA_from_scratch.principal_directions[j]
                            
    
    list_variance_along_prin_dir.append(np.real(variance_along_prin_dir))
    
    
print('\n\nCovariance of data projected on the first k principal directions:\n', list_variance_along_prin_dir)
print('\n\nEigenvalues by PCA_from_scratch class\n:', np.real(PCA_from_scratch.variance_along_principal_direction))


print('\n', '*'*40, '\n')


#%% Significative plot

fig, ax = plt.subplots(1, 2, figsize = [20,6])

factor_1 = 'Zest'
factor_2 = 'Hope'

ax[0].scatter(df[factor_1], df[factor_2], s = 5, c = 'dodgerblue')
ax[0].set_xlabel(factor_1)
ax[0].set_ylabel(factor_2)


# Considering PC1, PC2, X1, X2, V1, V2 be column vectors --> [PC1 PC2] = [X1 X2][V1 V2]
# PC1_1 = X1_1 * V1_1 + X2_1 * V1_2
# PC1_2 = X1_2 * V1_1 + X2_2 * V1_2
#...
# PC1_n = X1_n * V1_1 + X2_n * V1_2
# Projection of point P = [X1_1, X2_1] along V1 = [V1_1, V1_2] is np.dot(P, V1) = PC1_1
# if I multiply by vector V1 I obtain the vector np.dot(P, V1)*V1 = PC1_1*V1

for j in range(df.shape[0]):
    ax[0].scatter( df_PCA_from_scratch['PC1'][j] * principal_direction_1[factor_1],
                   df_PCA_from_scratch['PC1'][j] * principal_direction_1[factor_2],
                   s = 5, c = 'red')
    
min_asc = min(ax[0].get_xticks())
max_asc = max(ax[0].get_xticks())

q = 0
m = principal_direction_1[factor_2] / principal_direction_1[factor_1]
min_ord = m*min_asc + q
max_ord = m*max_asc + q

ax[0].plot([min_asc, max_asc], [min_ord, max_ord], linestyle = '--', c = 'limegreen')
    
ax[1].scatter(df_PCA_from_scratch['PC1'], df_PCA_from_scratch['PC2'], s = 5, c = 'dodgerblue')
ax[1].set_xlabel('PC1')
ax[1].set_ylabel('PC2')

plt.show()

print('\n', '*'*40, '\n')

    
#%% 3D plot

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(df_PCA_from_scratch['PC1'],
           df_PCA_from_scratch['PC2'],
           df_PCA_from_scratch['PC3'], 
           c = df_PCA_from_scratch['PC1'], cmap = 'viridis')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Points in space spanned by <PC1, PC2, PC3>')

plt.show()

print('\n', '*'*40, '\n')

#%%BIPLOT
# PCs = X @ V, so X = PCs @ V_T. Considering only 2 PCs PC1 and PC2:
# PC1 @ V1 + PC2 @ V2 = X and for each factor Xj
# Xj = y1 * V1_j + y2 * V2_j
# So we can write the original factors as linear combinations of the principal
# components basis, having the principal_directions' coefficients as weights

def biplot(PC, principal_directions, list_variable_names, idx_1 = 0, idx_2 = 1):
    
    # idx_1, idx_2 are indexes of the PCs to be plotted (es. PC1 and PC_3)
    
    PC_1 = PC[:, idx_1] # first principal component
    PC_2 = PC[:, idx_2] # second principal component
    
    PC_1 = PC_1/(max(PC_1) - min(PC_1))
    PC_2 = PC_2/(max(PC_2) - min(PC_2))
    
    
    plt.scatter(PC_1, PC_2, s = 5, c = PC_1, cmap = 'viridis')
    
    
    n = principal_directions.shape[0] # it's equal to the number of original factors
    
    for j in range(n):
        plt.arrow( x = 0, y = 0, # coordinates of the arrow base
                   dx = principal_directions[j, idx_1], # PC1 coordinate of arrow
                   dy = principal_directions[j, idx_2], # PC2 coordinate of arrow
                   color = 'red', head_width = 0.01, lw = 0.4)
        
        plt.text( principal_directions[j, idx_1]*1.1, principal_directions[j, idx_2]*1.1,
                  list_variable_names[j], color = 'red', fontsize = 8)
        
    plt.xlabel('PC_' + str(idx_1 + 1))
    plt.ylabel('PC_' + str(idx_2 + 1))
    
    return(None)
    

biplot(PC = PCs, 
       # with the transposition the eigenvectors V are on columns
       principal_directions = np.real(PCA_from_scratch.principal_directions.T),
       list_variable_names = cols,
       idx_1 = 0 , idx_2 = 1)

plt.savefig('A', dpi = 500)

plt.show()

print('\n', '*'*40, '\n')

#%%
