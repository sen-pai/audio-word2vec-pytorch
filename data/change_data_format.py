import numpy as np
import pickle

#load
with open('mfccs.pkl', 'rb') as f:
    data = pickle.load(f)
    
data_set_as_tuples = []

for i in data.keys():
    for j in data[i].keys():
        for word, mfcc in data[i][j].items():
            data_set_as_tuples.append((word,mfcc))
            
data_set_as_tuples = np.array(data_set_as_tuples)

#save as npy
np.save('mfcc_with_word.npy', data_set_as_tuples)
