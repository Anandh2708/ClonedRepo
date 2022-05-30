import pandas as pd
import numpy as np
from Analysis import Accuracy as ac
import relativePath as rp
import os

def compute_weighted_average(weights):
    predicted = []

    lstm_p = []
    lstm_interest_p = []
    lstm_crude_p = []
    lstm_hybrid_p = []

    for i in lstm['Prediction']:
        lstm_p.append(i)
    for i in lstm_interest['Prediction']:
        lstm_interest_p.append(i)
    for i in lstm_crude['Prediction']:
        lstm_crude_p.append(i)
    for i in lstm_hybrid['Prediction']:
        lstm_hybrid_p.append(i)

    for i in range(len(lstm_p)):
        temp = lstm_p[i]*weights[0]
        temp = temp + lstm_crude_p[i]*weights[1]
        temp = temp + lstm_interest_p[i]*weights[2]
        temp = temp + lstm_hybrid_p[i]*weights[3]
        predicted.append(temp)

    lstm_weighted = lstm
    lstm_weighted['Prediction'] = predicted
    #return lstm_weighted
    filename = os.path.join(rp.dirname, 'Medium/LSTM_weighted.csv')
    lstm_weighted.to_csv(filename, index=False)
    #lstm_weighted.to_csv('C:/Users/Gomathinayagam/PycharmProjects/StockOverflowR2/Medium/LSTM_weighted.csv', index=False)


#path = 'C:/Users/Gomathinayagam/PycharmProjects/StockOverflowR2/Medium/'
#path = os.path.join(rp.dirname, 'Medium')
#os.path.join(rp.dirname, 'Medium/LSTM.csv')

#lstm = pd.read_csv(path+'LSTM.csv')
lstm = pd.read_csv(os.path.join(rp.dirname, 'Medium/LSTM.csv'))
#lstm_crude = pd.read_csv(path+'LSTM_crude.csv')
lstm_crude = pd.read_csv(os.path.join(rp.dirname, 'Medium/LSTM_crude.csv'))
#lstm_interest = pd.read_csv(path+'LSTM_interest.csv')
lstm_interest = pd.read_csv(os.path.join(rp.dirname, 'Medium/LSTM_interest.csv'))
#lstm_hybrid = pd.read_csv(path+'LSTM_hybrid.csv')
lstm_hybrid = pd.read_csv(os.path.join(rp.dirname, 'Medium/LSTM_hybrid.csv'))
#lstm_weighted = pd.read_csv(path+'LSTM_weighted.csv')
lstm_weighted = pd.read_csv(os.path.join(rp.dirname, 'Medium/LSTM_weighted.csv'))

lstm_accuracy = ac.print_accuracy1()
lstm_accuracy_1 = ac.print_accuracy(7)
lstm_accuracy_2 = ac.print_accuracy(30)
lstm_accuracy_3 = ac.print_accuracy(180)
lstm_accuracy_4 = ac.print_accuracy(300)

#news = pd.read_csv(path+'news.csv')[0:20]
news = pd.read_csv(os.path.join(rp.dirname, 'Medium/news.csv'))[0:20]
#print(news)

#next_data = pd.read_csv(path+'next_day.csv')
next_data = pd.read_csv(os.path.join(rp.dirname, 'Medium/next_day.csv'))

next_day = next_data.values.tolist()

'''
tomorrow = []
tomorrow.append(next_day[0][0])#regular
tomorrow.append(next_day[1][0])#crude
tomorrow.append(next_day[2][0])#interest
tomorrow.append(next_day[3][0])#news
'''
tomorrow = next_day[0][0]*0.24+next_day[1][0]*0.62+next_day[2][0]*0.03+next_day[3][0]*0.11 #hybrid
temp = []
temp.append(tomorrow)
price_today = lstm['Close'].tolist()[-1]
change_percent = np.round(100 - (price_today * 100) / tomorrow, 2)
temp.append(change_percent)
next_day.append(temp)

#print(next_day)

#print(tomorrow)
#weights = [0.24,0.62,0.03,0.11]


#compute_weighted_average()
#16878.391250   16865

