'''
import ntpath
import os

import pandas as pd

import relativePath as rp
filename = os.path.join(rp.dirname, 'scratch/testSample.csv')
a = [12,34,5]
a = pd.DataFrame(a)
a.to_csv(filename,index=False)
print(filename)
'''
import relativePath as rp
import os

filename = os.path.join(rp.dirname, 'Medium/next_day.csv')

model_path = os.path.join(rp.dirname, 'Live/Models')

filename = os.path.join(rp.dirname, 'Medium/LSTM.csv')
save.to_csv(filename,index=False)