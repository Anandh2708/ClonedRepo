from Medium import API as api
import pandas as pd

def measure_accuracy(df, days):
    #days = 50
    actual = df['Close']
    predicted = df['Prediction']

    accuracy = []
    for i in range(len(actual)-days,len(actual)):
        temp = min(actual[i],predicted[i])/max(actual[i],predicted[i])
        temp = temp*100
        accuracy.append(temp)
    total_accuracy = sum(accuracy)/len(accuracy)
    return round(total_accuracy,2)

def print_accuracy(days):
    accuracy = {}
    #days = len(api.lstm_crude)

    #print("crude : ",measure_accuracy(api.lstm_crude, len(api.lstm_crude)))
    accuracy['crude'] = measure_accuracy(api.lstm_crude, days)
    #print("regular : ",measure_accuracy(api.lstm, len(api.lstm)))
    accuracy['regular'] = measure_accuracy(api.lstm, days)
    #print("interest : ",measure_accuracy(api.lstm_interest, len(api.lstm_interest)))
    accuracy['interest'] = measure_accuracy(api.lstm_interest,days)
    #print("hybrid : ",measure_accuracy(api.lstm_hybrid, len(api.lstm_hybrid)))
    accuracy['hybrid'] = measure_accuracy(api.lstm_hybrid, days)
    #print("weighted : ",measure_accuracy(api.lstm_weighted, len(api.lstm_weighted)))
    accuracy['weighted'] = measure_accuracy(api.lstm_weighted, days)
    return accuracy

def print_accuracy1():
    accuracy = {}
    #print("crude : ",measure_accuracy(api.lstm_crude, len(api.lstm_crude)))
    accuracy['crude'] = measure_accuracy(api.lstm_crude, len(api.lstm_crude))
    #print("regular : ",measure_accuracy(api.lstm, len(api.lstm)))
    accuracy['regular'] = measure_accuracy(api.lstm, len(api.lstm))
    #print("interest : ",measure_accuracy(api.lstm_interest, len(api.lstm_interest)))
    accuracy['interest'] = measure_accuracy(api.lstm_interest,len(api.lstm_interest))
    #print("hybrid : ",measure_accuracy(api.lstm_hybrid, len(api.lstm_hybrid)))
    accuracy['hybrid'] = measure_accuracy(api.lstm_hybrid, len(api.lstm_hybrid))
    #print("weighted : ",measure_accuracy(api.lstm_weighted, len(api.lstm_weighted)))
    accuracy['weighted'] = measure_accuracy(api.lstm_weighted, len(api.lstm_weighted))
    return accuracy


#print(print_accuracy())

#-----------------------------------------------

#path = 'C:/Users/Gomathinayagam/PycharmProjects/StockOverflowR2/Medium/'
'''
for i in range(100):
    for j in range(100):
        for k in range(100):
            for l in range(100):
                if(i+j+k+l == 100):
                    temp = []
                    temp.append(i/100)
                    temp.append(j/100)
                    temp.append(k/100)
                    temp.append(l/100)
                    current = api.compute_weighted_average(temp)
                    row = []
                    row.append(temp)
                    row.append(measure_accuracy(current, len(current)))
                    brute.append(row)
                    print(temp, measure_accuracy(current, len(current)))

brute = pd.DataFrame(brute)'''
#brute = []
#brute.to_csv("brute.csv")
