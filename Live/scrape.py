#import what we need
import re

from requests_html import HTMLSession
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import relativePath as rp
import os

remove = ['you','use','un','two','to','turns','their','says','on','of','wants','pen']

#support_path = 'C:\\Users\Gomathinayagam\PycharmProjects\StockOverflowR2\Live'
support_path = os.path.join(rp.dirname, 'Live')

def news_scrap():
    session = HTMLSession()

    # use session to get the page
    r = session.get('https://news.google.com/topstories')

    # render the html, sleep=1 to give it a second to finish before moving on. scrolldown= how many times to page down on the browser, to get more results. 5 was a good number here
    r.html.render(sleep=1, scrolldown=5, timeout=20)

    # find all the articles by using inspect element and create blank list
    articles = r.html.find('article')
    newslist = []

    # loop through each article to find the title and link. try and except as repeated articles from other sources have different h tags.
    for item in articles:
        try:
            newsitem = item.find('h3', first=True)
            title = newsitem.text
            link = newsitem.absolute_links
            newsarticle = {
                'title': title,
                'link': link
            }
            newslist.append(newsarticle)
        except:
            pass

    '''                                     
    # print the length of the list
    print(len(newslist))
    for i in newslist:
        print(i['title'])
    '''

    # convert dictionary to list
    live_news = []
    for i in newslist:
        live_news.append(i['title'])
    #print(live_news)

    # Create a Vectorizer Object
    vectorizer = CountVectorizer()

    vectorizer.fit(live_news)

    # Printing the identified Unique words along with their indices
    #print("Vocabulary: ", vectorizer.vocabulary_)
    c = []
    for i in vectorizer.vocabulary_:
        temp = []
        temp.append(i)
        if i in remove:
            continue
        temp.append(vectorizer.vocabulary_.get(i))
        c.append(temp)
    #print(c)
    c.sort(key=lambda x:x[1],reverse=True)
    c = pd.DataFrame(c)
    #path = 'C:/Users/Gomathinayagam/PycharmProjects/StockOverflowR2/Medium/'
    # c.to_csv(path+'news.csv',index=False)

    path = os.path.join(rp.dirname, 'Medium/news.csv')
    c.to_csv(path, index=False)
    #print(c)

    '''
    # Encode the Document
    vector = vectorizer.transform(live_news)

    # Summarizing the Encoded Texts
    print("Encoded Document is:")
    print(vector.toarray())
    '''



    return live_news


# Function for web scraping
class News_Prediction :

    # getting live news headline and store it in list and pre-processing
    live_headline = news_scrap()
    # print(live_headline)

    headlines = ''

    for i in live_headline:
        headlines += i

    headlines = re.sub("[^A-Za-z]", " ", headlines)

    headlines = headlines.lower()

    # Function to train a random forest algorithm
    countvector = CountVectorizer(ngram_range=(2, 2))
    randomclassifier = RandomForestClassifier(n_estimators=200, criterion='entropy')

    def random_train(self):
        df = pd.read_csv(support_path+'\Data.csv', encoding="ISO-8859-1")

        train = df[df['Date'] < '20150101']
        # test = df[df['Date'] > '20141231']

        # Removing punctuations
        data = train.iloc[:, 2:27]
        data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
        # print(data)
        # Renaming column names for ease of access
        list1 = [i for i in range(25)]
        new_Index = [str(i) for i in list1]
        data.columns = new_Index

        # Convertng headlines to lower case
        for index in new_Index:
            data[index] = data[index].str.lower()

        ' '.join(str(x) for x in data.iloc[1, 0:25])

        headlines = []
        for row in range(0, len(data.index)):
            headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))

        #print(headlines[0])

        ## implement BAG OF WORDS
        traindataset = self.countvector.fit_transform(headlines)

        # implement RandomForest Classifier
        self.randomclassifier.fit(traindataset, train['Label'])

    def news_prediction(self):
        self.random_train()
    # Predicting live news data
        hl = []
        hl.append(self.headlines)
        test_dataset = self.countvector.transform(hl)
        predictions = self.randomclassifier.predict(test_dataset)
        return predictions

n = News_Prediction()
n.news_prediction()