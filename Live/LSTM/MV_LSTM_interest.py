import csv
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense
from datetime import date, timedelta
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from Live.scrape import News_Prediction
import relativePath as rp
import os

#model_path = "C:\\Users\Gomathinayagam\PycharmProjects\StockOverflowR2\Live\Models"
model_path = os.path.join(rp.dirname, 'Live/Models')

class MV_LSTM_interest:
    # Time Frame
    today = date.today()
    date_today = today.strftime("%Y-%m-%d")
    date_start = '2012-02-01'

    # Security ID
    stockname = 'NIFTY 50'
    nifty_id = '^NSEI'
    crude_id = 'CL=F'

    epochs = 50
    sequence_length = 50
    train_data_len = 0

    scaler = MinMaxScaler()

    df = yf.download(nifty_id, start=date_start, end=date_today)

    # adding interest rate to data frame
    #df.to_csv('file1.csv')
    #df.to_csv("C:/Users/Gomathinayagam/PycharmProjects/StockOverflowR2/Live/file1.csv", 'w')
    filename = os.path.join(rp.dirname, 'Live/file1.csv')
    df.to_csv(filename, 'w')

    interest = []
    # with open('ir_rbi.csv', 'r') as file:
    filename = os.path.join(rp.dirname, 'Live/ir_rbi.csv')
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row[1]:
                continue;
            else:
                for i in range(21):
                    interest.append(row[1])
    interest = interest[0:len(df)]
    df['interest'] = interest
    # ----------------------------------

    crude = yf.download(crude_id, start=date_start, end=date_today)

    data = []
    data_filtered = []

    x_test = []
    y_text = []

    date_index = []

    def __init__(self, epochs, sequence_length):
        self.epochs = epochs
        self.sequence_length = sequence_length

    def clt_LSTM_Model(self):

        self.crude.index = pd.to_datetime(self.crude.index)
        crude_price = []

        for i in range(len(self.df)):
            date = str(self.df.index[i])
            date = date[0:10]
            try:
                crude_price.append(self.crude.loc[date]['Adj Close'])
            except:
                crude_price.append(crude_price[-1])

        self.df['Crude Oil'] = crude_price

        # Preprocessing and Feature Selection

        # Indexing Batches
        train_df = self.df.sort_values(by=['Date']).copy()
        self.date_index = train_df.index
        train_df = train_df.reset_index(drop=True).copy()

        # List of considered Features
        FEATURES = ['High', 'Low', 'Open', 'Close', 'Volume', 'interest']

        # Create the dataset with features and filter the data to the list of FEATURES
        self.data = pd.DataFrame(train_df)
        self.data_filtered = self.data[FEATURES]

        # We add a prediction column and set dummy values to prepare the data for scaling
        data_filtered_ext = self.data_filtered.copy()
        data_filtered_ext['Prediction'] = data_filtered_ext['Close']

        # Get the number of rows in the data
        nrows = self.data_filtered.shape[0]

        # Convert the data to numpy values
        np_data_unscaled = np.array(self.data_filtered)
        np_data = np.reshape(np_data_unscaled, (nrows, -1))
        # print(np_data.shape)

        # Transform the data by scaling each feature to a range between 0 and 1
        np_data_scaled = self.scaler.fit_transform(np_data_unscaled)

        # Creating a separate scaler that works on a single column for scaling predictions
        scaler_pred = MinMaxScaler()
        df_Close = pd.DataFrame(data_filtered_ext['Close'])
        np_Close_scaled = scaler_pred.fit_transform(df_Close)

        # Prediction Index
        index_Close = self.data.columns.get_loc("Close")

        # Split the training data into train and train data sets
        # As a first step, we get the number of rows to train the model on 80% of the data
        self.train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

        # Create the training and test data
        train_data = np_data_scaled[0:self.train_data_len, :]
        test_data = np_data_scaled[self.train_data_len - self.sequence_length:, :]

        def partition_dataset(sequence_length, data):
            x, y = [], []
            data_len = data.shape[0]
            for i in range(sequence_length, data_len):
                x.append(data[i - sequence_length:i, :])
                y.append(data[i, index_Close])

                # Convert the x and y to numpy arrays
            x = np.array(x)
            y = np.array(y)
            return x, y

        # Generate training data and test data
        x_train, y_train = partition_dataset(self.sequence_length, train_data)
        self.x_test, self.y_test = partition_dataset(self.sequence_length, test_data)

        # Model Training

        # Configure the neural network model
        model = Sequential()

        # Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables

        n_neurons = x_train.shape[1] * x_train.shape[2]
        print(n_neurons, x_train.shape[1], x_train.shape[2])
        model.add(LSTM(n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(n_neurons, return_sequences=False))
        model.add(Dense(5))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Training the model

        batch_size = 16
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=self.epochs,
                            validation_data=(self.x_test, self.y_test)
                            )
        #model.save('MVLSTM_interest.h5')
        model.save(model_path+'\MVLSTM_interest.h5')

    # -------------------------------------------------------

    def load_Test_Model(self):

        self.crude.index = pd.to_datetime(self.crude.index)

        crude_price = []

        for i in range(len(self.df)):
            date = str(self.df.index[i])
            date = date[0:10]
            try:
                crude_price.append(self.crude.loc[date]['Adj Close'])
            except:
                crude_price.append(crude_price[-1])

        self.df['Crude Oil'] = crude_price

        # Preprocessing and Feature Selection

        # Indexing Batches
        train_df = self.df.sort_values(by=['Date']).copy()

        # We safe a copy of the dates index, before we need to reset it to numbers
        date_index = train_df.index

        # We reset the index, so we can convert the date-index to a number-index
        train_df = train_df.reset_index(drop=True).copy()
        # train_df.head(5)

        # List of considered Features
        FEATURES = ['High', 'Low', 'Open', 'Close', 'Volume', 'interest']

        # print('FEATURE LIST')
        # print([f for f in FEATURES])

        # Create the dataset with features and filter the data to the list of FEATURES
        data = pd.DataFrame(train_df)
        data_filtered = data[FEATURES]

        # We add a prediction column and set dummy values to prepare the data for scaling
        data_filtered_ext = data_filtered.copy()
        data_filtered_ext['Prediction'] = data_filtered_ext['Close']

        # Print the tail of the dataframe
        # data_filtered_ext.tail()

        # Get the number of rows in the data
        nrows = data_filtered.shape[0]

        # Convert the data to numpy values
        np_data_unscaled = np.array(data_filtered)
        np_data = np.reshape(np_data_unscaled, (nrows, -1))
        # print(np_data.shape)

        # Transform the data by scaling each feature to a range between 0 and 1
        scaler = MinMaxScaler()
        np_data_scaled = scaler.fit_transform(np_data_unscaled)

        # Creating a separate scaler that works on a single column for scaling predictions
        scaler_pred = MinMaxScaler()
        df_Close = pd.DataFrame(data_filtered_ext['Close'])
        np_Close_scaled = scaler_pred.fit_transform(df_Close)

        # Set the sequence length - this is the timeframe used to make a single prediction

        # Prediction Index
        index_Close = data.columns.get_loc("Close")

        # Split the training data into train and train data sets
        # As a first step, we get the number of rows to train the model on 80% of the data
        train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

        # Create the training and test data
        train_data = np_data_scaled[0:train_data_len, :]
        test_data = np_data_scaled[train_data_len - self.sequence_length:, :]

        # The RNN needs data with the format of [samples, time steps, features]
        # Here, we create N samples, sequence_length time steps per sample, and 6 features
        def partition_dataset(sequence_length, data):
            x, y = [], []
            data_len = data.shape[0]
            for i in range(sequence_length, data_len):
                x.append(data[i - sequence_length:i, :])  # contains sequence_length values 0-sequence_length * columsn
                y.append(
                    data[i, index_Close])  # contains the prediction values for validation,  for single-step prediction

            # Convert the x and y to numpy arrays
            x = np.array(x)
            y = np.array(y)
            return x, y

        # Generate training data and test data
        x_train, y_train = partition_dataset(self.sequence_length, train_data)
        x_test, y_test = partition_dataset(self.sequence_length, test_data)

        # Model Training
        #model = load_model('MVLSTM_interest.h5')
        model = load_model(model_path+'\MVLSTM_interest.h5')

        y_pred_scaled = model.predict(x_test)

        # Unscale the predicted values
        y_pred = scaler_pred.inverse_transform(y_pred_scaled)
        y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

        # Mean Absolute Error (MAE)
        MAE = mean_absolute_error(y_test_unscaled, y_pred)
        # print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

        # Mean Absolute Percentage Error (MAPE)
        MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred) / y_test_unscaled))) * 100
        # print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

        # Median Absolute Percentage Error (MDAPE)
        MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred) / y_test_unscaled))) * 100
        # print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

        # The date from which on the date is displayed
        display_start_date = pd.Timestamp('today') - timedelta(days=500)

        # Add the date column
        data_filtered_sub = data_filtered.copy()
        data_filtered_sub['Date'] = date_index

        # Add the difference between the valid and predicted prices
        train = data_filtered_sub[:train_data_len + 1]
        valid = data_filtered_sub[train_data_len:]
        valid.insert(1, "Prediction", y_pred.ravel(), True)
        valid.insert(1, "Difference", valid["Prediction"] - valid["Close"], True)

        # Zoom in to a closer timeframe
        valid = valid[valid['Date'] > display_start_date]
        train = train[train['Date'] > display_start_date]
        # -----------------------------------------------------------------
        # print("-------- valid -------")
        # print(valid)
        # print("\n")
        select = ['Date', 'Close', 'Prediction']
        save = valid
        save = save[select]
        # print(save)
        #save.to_csv('C:/Users/Gomathinayagam/PycharmProjects/StockOverflowR2/Medium/LSTM_interest.csv',index=False)
        filename = os.path.join(rp.dirname, 'Medium/LSTM_interest.csv')
        save.to_csv(filename, index=False)

        # -----------------------------------------------------------------
        # Predict Next Day's Price

        df_temp = self.df[-self.sequence_length:]
        new_df = df_temp.filter(FEATURES)

        N = self.sequence_length

        # Get the last N day closing price values and scale the data to be values between 0 and 1
        last_N_days = new_df[-self.sequence_length:].values
        last_N_days_scaled = scaler.transform(last_N_days)

        # Create an empty list and Append past N days
        X_test_new = []
        X_test_new.append(last_N_days_scaled)

        # Convert the X_test data set to a numpy array and reshape the data
        pred_price_scaled = model.predict(np.array(X_test_new))
        pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))

        # Print last price and predicted price for the next day
        price_today = np.round(new_df['Close'][-1], 2)
        predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
        change_percent = np.round(100 - (price_today * 100) / predicted_price, 2)

        plus = '+';
        minus = ''
        #print(f'The close price for {self.stockname} at {self.today} was {price_today}')
        #print(
        #    f'The predicted close price is {predicted_price} ({plus if change_percent > 0 else minus}{change_percent}%)')
        result = []
        # print(predicted_price)
        # print(change_percent)
        result.append(predicted_price)
        result.append(change_percent)
        return result
