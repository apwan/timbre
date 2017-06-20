import json
from pprint import pprint
import librosa
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
import warnings
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
import time

warnings.filterwarnings("ignore")

def prepare_features(x, fs, dataitem):
    # Set the hop length; at 16000 Hz, 512 samples ~= 20ms
    if 'zero_crossings_ave' not in dataitem.keys():
        zero_crossings = librosa.zero_crossings(x)
        zero_crossings_ave = np.mean(zero_crossings)
        dataitem['zero_crossings_ave'] = [zero_crossings_ave]
    if 'mfcc' not in dataitem.keys():
        mfccs = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=40)
        mfccs_ave = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        dataitem['mfccs'] = mfccs
        dataitem['mfccs_ave'] = mfccs_ave
        dataitem['mfccs_std'] = mfccs_std
    if 'spectral_centroids' not in dataitem.keys():
        spectral_centroids = librosa.feature.spectral_centroid(x, sr=fs).ravel()
        spectral_centroids_ave = np.mean(spectral_centroids)
        dataitem['spectral_centroids'] = spectral_centroids
        dataitem['spectral_centroids_ave'] = [spectral_centroids_ave]
    return dataitem

def extract_features(dataitem):
    feat_names=['zero_crossings_ave','mfccs_ave','mfccs_std','spectral_centroids_ave']
    feat=[]
    for name in feat_names:
        feat.extend(dataitem[name])
    return feat

'''
with open('nsynth-valid/examples.json') as data_file:
	valid_data = json.load(data_file)

with open('nsynth-test/examples.json') as data_file:
    test_data = json.load(data_file)
'''

test_featured_file = 'nsynth-valid/examples_featured.pkl'
valid_featured_file = 'nsynth-test/examples_featured.pkl'

start = time.time()
if False:
    test_data = json.load(open('nsynth-test/examples.json'))
    valid_data = json.load(open('nsynth-valid/examples.json'))
else:
    test_data = pickle.load(open(test_featured_file,'rb'))
    valid_data = pickle.load(open(valid_featured_file,'rb'))
print 'Finished loading in', time.time() - start

def test_data_worker(pair):
    i, item = pair
    if i % 10 == 0:
        print i
    filename = 'nsynth-test/audio/' + item + '.wav'
    y, sr = librosa.load(filename)
    return prepare_features(y, sr, test_data[item])

def valid_data_worker(pair):
    i, item = pair
    if i % 10 == 0:
        print i
    filename = 'nsynth-valid/audio/' + item + '.wav'
    y, sr = librosa.load(filename)
    return prepare_features(y, sr, valid_data[item])    

#pool = Pool(4)

'''
start = time.time()
results = pool.map(test_data_worker, enumerate(test_data.keys()))
for num, item in enumerate(test_data.keys()):
    test_data[item] = results[num]
print 'Finished in', time.time() - start
pickle.dump(test_data, open(test_featured_file, 'wb'))

start = time.time()
results = pool.map(valid_data_worker, enumerate(valid_data.keys()))
for num, item in enumerate(valid_data.keys()):
    valid_data[item] = results[num]
pickle.dump(valid_data, open(valid_featured_file, 'wb'))
print 'Finished in', time.time() - start
'''

X = []
Y = []
for num, item in enumerate(test_data.keys()):
    #if num % 1000 == 0:
    #    print num
    X.append(extract_features(test_data[item]))
    Y.append(test_data[item]['instrument_source'])
X = np.array(X)
Y = np.array(Y)
print X.shape, Y.shape

valid_X = []
valid_Y = []
for num, item in enumerate(valid_data.keys()):
    #if num % 1000 == 0:
    #    print num
    valid_X.append(extract_features(valid_data[item]))
    valid_Y.append(valid_data[item]['instrument_source'])
valid_X = np.array(valid_X)
valid_Y = np.array(valid_Y)

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
valid_X = scaler.transform(valid_X)

#clf = SVC()
#clf.fit(X, Y)

'''
from sklearn.cluster import KMeans
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(valid_X)
preds = kmeans.predict(valid_X)

pred_by_label = np.zeros((n_clusters,11))
for pred, label in zip(preds, valid_Y):
    pred_by_label[pred,label] += 1

translation = np.argmax(pred_by_label, axis=1)
print translation

Predicted_Y = [translation[pred] for pred in preds]
print np.mean(np.asarray(valid_Y) == Predicted_Y)
'''

'''
from sklearn.neural_network import MLPClassifier,MLPRegressor
clf = MLPClassifier(hidden_layer_sizes=(64,16,2,16),activation='tanh')
Y_one_hot=np.array([np.eye(11)[label] for label in Y])
clf.fit(X,Y_one_hot)
Predicted_Y=clf.predict_proba(valid_X).argmax(axis=1)
print np.mean(np.asarray(valid_Y) == Predicted_Y)
'''

'''
from mlp import MLP
clf=MLP()
clf.fit(X,Y)
Predicted_Y = clf.predict(valid_X)
print np.mean(np.asarray(valid_Y) == Predicted_Y)
'''

from mlp import MLPRegressor
clf=MLPRegressor()
clf.fit(X)
X_transformed=clf.transform(X)
print X_transformed.shape

for i in range(11):
    indices = Y == i
    plt.scatter(X_transformed[indices][:,0],X_transformed[indices][:,1],c=np.random.rand(3),label=str(i))
plt.legend()
plt.show()





