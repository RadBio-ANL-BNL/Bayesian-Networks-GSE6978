'''
=====
Distributed by: Computational Science Initiative, Brookhaven National Laboratory (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
=====
'''
import collections
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif


#####################################
# feature selection
#####################################
sepLineS = '-'*37
sepLineE = '-'*37+'\n'

# load probe-level data
data = pd.read_csv('data/GSE6978_full.csv')
dose = data['Dose']

# define label
labels = []
cG0, cG05, cG005 = 0, 0, 0
for item in dose:
    if item == '0Gy':
        labels.append(0)
        cG0 += 1
    else:
        labels.append(1)
        if item == '0.5Gy':
            cG05 += 1
        else:
            cG005 += 1

print(sepLineS)
print('number of exposed 0Gy is ', cG0)
print('number of exposed 0.5Gy is ', cG05)
print('number of exposed 0.05Gy is ', cG005)
print(sepLineE)

# determine the idces
idx0 = []
idx05 = []
idx005 = []
for idx, item in enumerate(dose):
    if item == '0Gy':
        idx0.append(idx)
    elif item == '0.5Gy':
        idx05.append(idx)
    elif item == '0.05Gy':
        idx005.append(idx)

# determine the training and test idces
train_idx0, test_idx0 = train_test_split(idx0, test_size=5)
train_idx05, test_idx05 = train_test_split(idx05, test_size=5)
train_idx005, test_idx005 = train_test_split(idx005, test_size=5)
np.savez('data/idces_train_test', train_idx0=train_idx0, train_idx05=train_idx05, train_idx005=train_idx005, test_idx0=test_idx0, test_idx05=test_idx05, test_idx005=test_idx005)

# formulate inputs
data = np.transpose(data.to_numpy())

train_x0 = data[:, train_idx0]
train_x05 = data[:, train_idx05]
train_x005 = data[:, train_idx005]

test_x0 = data[:, test_idx0]
test_x05 = data[:, test_idx05]
test_x005 = data[:, test_idx005]

X_train = np.concatenate((np.concatenate((train_x0, train_x05), axis=1), train_x005), axis=1)
X_test = np.concatenate((np.concatenate((test_x0, test_x05), axis=1), test_x005), axis=1)
X = np.concatenate((X_train, X_test), axis=1)
X = np.transpose(X)

print(sepLineS)
print('The last column ', X[:,-1])
print('The second last column ', X[:,-2])
print('The third last column ', X[:,-3])
print(sepLineE)
X = X[:,:-3]

train_y0 = [0 for _ in range(len(train_idx0))]
train_y05 = [1 for _ in range(len(train_idx05))]
train_y005 = [1 for _ in range(len(train_idx005))]

test_y0 = [0 for _ in range(len(test_idx0))]
test_y05 = [1 for _ in range(len(test_idx05))]
test_y005 = [1 for _ in range(len(test_idx005))]

Y_train = np.concatenate((np.concatenate((train_y0, train_y05), axis=0), train_y005), axis=0)
Y_test = np.concatenate((np.concatenate((test_y0, test_y05), axis=0), test_y005), axis=0)
Y = np.concatenate((Y_train, Y_test), axis=0)

print(sepLineS)
print('The shape of X is', X.shape)
print('The shape of Y is', Y.shape)
print(sepLineE)

# select features and build classifier
numFea = 20
selector = SelectKBest(f_classif, k=numFea)
X_new = selector.fit_transform(X, Y)
cols = selector.get_support(indices=True)
np.save('data/idces_{}_features_cols.npy'.format(numFea), cols)

X_train, X_test = X_new[:99], X_new[99:]
y_train, y_test = Y[:99], Y[99:]
np.savez('data/probe_data_{}_features'.format(numFea), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print(sepLineS)
print('{} of features are used to build the model'.format(numFea))
print('Accuracy = {}'.format(score))
print(sepLineE)


#####################################
# binary transformation
#####################################
data = np.load('data/probe_data_20_features.npz', allow_pickle=True)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(sepLineS)
print('X train shape is ', X_train.shape)
print('X test shape is ', X_test.shape)
print('y train shape is ', y_train.shape)
print('y test shape is ', y_test.shape)
print(sepLineE)

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

print(sepLineS)
print('X shape is ', X.shape)
print('y shape is ', y.shape)
print(sepLineE)

xMean = []
for idx in range(np.shape(X)[1]):
    xMean.append(np.mean(X[:,idx]))

for idxCol, item in enumerate(xMean):
    for idxRow in range(np.shape(X)[0]):
        if X[idxRow][idxCol] < item:
            X[idxRow][idxCol] = 0
        else:
            X[idxRow][idxCol] = 1
tab = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)

print(sepLineS)
print('Table shape is ', tab.shape)
print(sepLineE)

ncols = list(tab.shape)
dic = {'x'+str(idx+1):tab[:,idx] for idx in range(ncols[1])}
dic['y'] = dic.pop('x'+str(ncols[1]))

df = pd.DataFrame(dic) 
df.to_csv('data/binary_{}_features.csv'.format(np.shape(X)[1]))


#####################################
# convert genes
#####################################
data = pd.read_csv('data/GSE6978_full.csv')

numFea = 20
selectCols = np.load('data/idces_{}_features_cols.npy'.format(numFea))
columnNames = data.columns[selectCols]

print(sepLineS)
print('Selected column indces ', selectCols)
print('Selected column names ', columnNames)
print(sepLineE)

convTable = pd.read_csv('data/GSE6978_p2g.txt', delimiter='\t')
idCol = convTable['ID']
GeneSymbol = convTable['GeneSymbol']   # Gene symbol
Gene = convTable['GENE']               # Entrez GENE ID

print(sepLineS)
print('probe to gene conversion table shape', convTable.shape)
print('Gene symbol shape', GeneSymbol.shape)
print('Entrez GENE ID shape', Gene.shape)
print(sepLineE)

rowNums = []
for columnName in columnNames:
    for idx in range(len(idCol)):
        if columnName == idCol[idx]:
            rowNums.append(idx)

convetGeneSymbols = GeneSymbol[rowNums]
convetGenes = Gene[rowNums]

print(sepLineS)
print('converted Gene symbol ', convetGeneSymbols)
print('converted Entrez GENE ID ',convetGenes)
print(sepLineE)


