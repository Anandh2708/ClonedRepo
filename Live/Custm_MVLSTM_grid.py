from Live.LSTM.MV_LSTM import MV_LSTM
from Live.LSTM.MV_LSTM_interest import MV_LSTM_interest
from Live.LSTM.MV_LSTM_Hybrid import MV_LSTM_Hybrid
from Live.LSTM.MV_LSTM_crude import MV_LSTM_crude
from Medium import API as api
import relativePath as rp
import os

#model_path = "C:\\Users\Gomathinayagam\PycharmProjects\StockOverflowR2\Live\Models"
model_path = os.path.join(rp.dirname, 'Live/Models')

def controller ():
    regular = MV_LSTM(50, 50)
    crude = MV_LSTM_crude(50, 50)
    interest = MV_LSTM_interest(50, 50)
    hybrid = MV_LSTM_Hybrid(50,50)

    #regular.clt_LSTM_Model()
    #crude.clt_LSTM_Model()
    #interest.clt_LSTM_Model()
    #hybrid.clt_LSTM_Model()

    result_Matrix = []
    result_Matrix.append(regular.load_Test_Model())
    result_Matrix.append(crude.load_Test_Model())
    result_Matrix.append(interest.load_Test_Model())
    result_Matrix.append(hybrid.load_Test_Model())
    weights = [0.24,0.62,0.03,0.11]
    #[0.24, 0.62, 0.03, 0.11]
    api.compute_weighted_average(weights)
    return result_Matrix



#print(controller())

