from django.shortcuts import render
from django.http import HttpResponse
import plotly.express as px
import pandas as pd
from Medium import API as api

# Create your views here.

HybridPrice = round(api.next_day[4][0],2)
HybridPercent = api.next_day[4][1]

def index(request):

    Nifty = {
        'HybridPrice' : HybridPrice,
        'HybridPercent':HybridPercent
    }

    return render(request,'index.html',{'Nifty': Nifty })

# Function to genarate graph
def generate_graph(df,graph_name):

    fig = px.line(df, x='Date', y=df.columns[1:3],
                  labels={
                      "value": "Nifty 50(₹)",
                      "variable": "Price",
                  },
                  title= graph_name,)
    graph = fig.to_html(config={'displaylogo': False, 'scrollZoom': True})
    return graph

def lstm(request):

    res_graph = {
        'lstm_graph' : generate_graph(api.lstm,"REGULAR LSTM"),
        'lstm_crude_graph': generate_graph(api.lstm_crude, "LSTM with Crude Oil price"),
        'lstm_interest_graph': generate_graph(api.lstm_interest, "LSTM with Interest Rate"),
        'lstm_news_graph': generate_graph(api.lstm_hybrid, "LSTM with News"),
        'weighted_lstm_graph': generate_graph(api.lstm_weighted,"Hybrid Prediction"),
        'lstm_value': api.next_day[0][0],
        'lstm_crude_value': api.next_day[1][0],
        'lstm_interest_value' :api.next_day[2][0],
        'lstm_news_value' : api.next_day[3][0],
        'lstm_hybrid_value': api.next_day[4][0],
        'lstm_percent': api.next_day[0][1],
        'lstm_crude_percent': api.next_day[1][1],
        'lstm_interest_percent': api.next_day[2][1],
        'lstm_news_percent': api.next_day[3][1],
        'lstm_hybrid_percent': api.next_day[4][1]
    }
    return render(request,'lstm.html',{'res_graph': res_graph})


def news(request):

    news_df = api.news
    fig = px.bar(news_df, x='0', y='1',
                  labels={
                      "0": "Word",
                      "1": "Count",
                  },
                  title="Words and it's Count from Live News", )
    news_graph = fig.to_html(config={'displaylogo': False, 'scrollZoom': True})
    return render(request,'news.html',{'news_graph' : news_graph})

def statistics(request):

    accuracy = {
        'LSTM': api.lstm_accuracy['regular'],
        'LSTMWITHCRUDE': api.lstm_accuracy['crude'],
        'LSTMWITHNEWS': api.lstm_accuracy['hybrid'],
        'LSTMWITHINTERESTRATE': api.lstm_accuracy['interest'],
        'WEIGHTEDLSTM': api.lstm_accuracy['weighted'],
        'LSTM1': api.lstm_accuracy_1['regular'],
        'LSTMWITHCRUDE1': api.lstm_accuracy_1['crude'],
        'LSTMWITHNEWS1': api.lstm_accuracy_1['hybrid'],
        'LSTMWITHINTERESTRATE1': api.lstm_accuracy_1['interest'],
        'WEIGHTEDLSTM1': api.lstm_accuracy_1['weighted'],
        'LSTM2': api.lstm_accuracy_2['regular'],
        'LSTMWITHCRUDE2': api.lstm_accuracy_2['crude'],
        'LSTMWITHNEWS2': api.lstm_accuracy_2['hybrid'],
        'LSTMWITHINTERESTRATE2': api.lstm_accuracy_2['interest'],
        'WEIGHTEDLSTM2': api.lstm_accuracy_2['weighted'],
        'LSTM3': api.lstm_accuracy_3['regular'],
        'LSTMWITHCRUDE3': api.lstm_accuracy_3['crude'],
        'LSTMWITHNEWS3': api.lstm_accuracy_3['hybrid'],
        'LSTMWITHINTERESTRATE3': api.lstm_accuracy_3['interest'],
        'WEIGHTEDLSTM3': api.lstm_accuracy_3['weighted'],
        'LSTM4': api.lstm_accuracy_4['regular'],
        'LSTMWITHCRUDE4': api.lstm_accuracy_4['crude'],
        'LSTMWITHNEWS4': api.lstm_accuracy_4['hybrid'],
        'LSTMWITHINTERESTRATE4': api.lstm_accuracy_4['interest'],
        'WEIGHTEDLSTM4': api.lstm_accuracy_4['weighted'],
    }

    return render(request,'statistics.html',{'accuracy' : accuracy})


def technical(request):
 return render(request,'technical.html')



