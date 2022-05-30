from Live import  Custm_MVLSTM_grid as grid
import pandas as pd
import relativePath as rp
import os

'''
class LSTM_op:

    LSTM = []
    LSTM_crude = []
    LSTM_hybrid = []
    LSTM_interest = []

    def __init__(self,predictions):
        self.LSTM = predictions[0]
        self.LSTM_crude = predictions[1]
        self.LSTM_interest = predictions[2]
        self.LSTM_hybrid = predictions[3]

LSTM = LSTM_op(grid.controller())
print(LSTM.LSTM_interest)
'''
next_day = grid.controller()
next_day = pd.DataFrame(next_day)
#next_day.to_csv('C:/Users/Gomathinayagam/PycharmProjects/StockOverflowR2/Medium/next_day.csv',index=False)

filename = os.path.join(rp.dirname, 'Medium/next_day.csv')
#print(filename)
next_day.to_csv(filename,index=False)
#Medium/next_day.csv