import pandas as pd
import math
import heapq

# useful to create the scraping of the pages
from bs4 import BeautifulSoup
import lxml
import requests

import os
import csv

# useful to create the bag of words
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from langdetect import detect

# libraries for the 4th question
import re
import operator
import matplotlib.pyplot as plt
import seaborn as sns

################################ RQ1 ################################

##### RQ1.1

def get_urls(initial_url, n_pages): # get the url of each single book
    UrlsFiles = open("urlpages.txt", "w")

    for i in range(1, n_pages+1):
        url = initial_url +str(i)

        page = requests.get(url)
        soup = BeautifulSoup(page.content, features='lxml')
        for a in soup.find_all('a', class_="bookTitle"):
            UrlsFiles.write(a.get('href')+'\n')

    UrlsFiles.close()

##### RQ1.2

def crawl_urls(n_pages, url_file):
    pathAncestor = os.path.join("./", "htmlpages") # Path
    os.mkdir(pathAncestor) # create the folder in the path
    
    for i in range(1,n_pages+1):
        os.makedirs(os.path.join(pathAncestor, 'page ' + str(i))) # create sequentially the folders interested
    
    UrlsFiles = open(url_file, "r") # open the file in "read (r)" mode

    headpart = "https://www.goodreads.com"
    counter_pages = 0
    counter_html = 0
    for x in UrlsFiles: # crawl each Urls associated to the book to be sure to download the corresponding html article
        if counter_html % 100 == 0: # check every hundred html page to change the folder where we insert the html article
            counter_pages = counter_pages + 1 

        counter_html = counter_html + 1

        subdirectory = pathAncestor + "/page " + str(counter_pages) # select the corresponding folder to insert the html article
        article_name = "/article_"+str(counter_html)+".html" # set the number of i-th book

        complete_path = subdirectory + article_name # insert the new complete path where create the html file
        with open(complete_path, "wb") as ip_file:
            link = headpart + x
            try:
                page = requests.get(link) # request the page
            except:
                with open("failureRequest.txt", "a") as err_file: # if we loose the request book, we put into a file the link doesn't download well, then we set the "urlpages.txt" with these link
                    err_file.write(link)
                    err_file.close()

            soup = BeautifulSoup(page.text, features='lxml')

            ip_file.write(soup.encode('utf-8'))
            ip_file.close()

    UrlsFiles.close()

##### RQ1.3

def scrap_book(tsv_writer, article): 
    global bookTitle, bookSeries, bookAuthors, ratingValue, ratingCount, reviewCount, Plot, NumberofPages, Published, Characters, Setting, Url; # set global variables to be sure that we consider into a scope this variables!
    
    with open(article, 'r', encoding="utf-8") as out_file: # for each html article downloaded scrape it!
        contents = out_file.read()
        soup = BeautifulSoup(contents, features="lxml") #parse the text
        
        # there are different excepts to be sure that if into the html article there isn't a information set it to empty string according to the professor requests
        
        # extract rating and review count
        try:
            ratings = soup.find_all('a', href="#other_reviews") #search the ratings in its
            rating_count = -1
            rating = -1
            for raiting in ratings:
                if raiting.find_all('meta', itemprop="ratingCount"):
                    ratingCount = raiting.text.replace('\n', '').strip().split(' ')[0].replace(',', '')
                elif raiting.find_all('meta', itemprop="reviewCount"):
                    reviewCount = raiting.text.replace('\n', '').strip().split(' ')[0].replace(',', '')
        except:
            ratingCount = " "
            reviewCount = " "
            
        # extract the book title
        try:
            bookTitle = soup.find_all('h1')[0].contents[0].replace('\n', '').strip()
        except:
            bookTitle = " "

        # extract the book authors
        try:
            bookAuthors = soup.find_all('span', itemprop='name')[0].contents[0]
        except:
            bookAuthors = " "

        # extract the book authors, we shoul FIX it.
        try:
            Plot = soup.find_all('div', id="description")[0].contents[3].text
            if detect(Plot) != "en":
                Plot = " "
        except:
            try:
                Plot = soup.find_all('div', id="description")[0].contents[1].text
                if detect(Plot) != "en":
                    Plot = " "
            except:
                Plot = " "
                

        # extract the date
        try:
            date = soup.find_all('div', id="details")[0].contents[3].text.replace('\n', '').strip().split()
            Published = date[1]+" "+date[2]+" "+date[3]
        except:
            Published = " "

        # Rating Value
        try:
            ratingValue = soup.find('span', itemprop="ratingValue").text.strip()
        except:
            ratingValue = " "

        # Number of pages
        try:
            NumberofPages = soup.find('span', itemprop="numberOfPages").text.split()[0]
        except:
            NumberofPages = " "

        # Title series
        try:
            bookSeries = soup.find_all('a', href= re.compile(r'/series/*'))[0].contents[0].strip()
        except:
            bookSeries = " "
            
        # Places
        try:
            Setting = []
            for places in soup.find_all('a', href= re.compile(r'/places/*')):
                Setting.append(places.text)
            Setting = ", ".join(Setting) if len(Setting)>=1 else " "
        except:
            Setting = " "

        # list of characters
        try:
            Characters = []
            for character in soup.find_all('a', href= re.compile(r'/characters/*') ):
                Characters.append(character.text)
            Characters = ", ".join(Characters) if len(Characters)>=1 else " "
        except:
            Characters = " "

        # extract the Url
        try:
            Url = soup.find_all('link')[0]["href"]
        except:
            Url = " "

        tsv_writer.writerow([bookTitle, bookSeries, bookAuthors, ratingValue, ratingCount, reviewCount, Plot, NumberofPages, Published, Characters, Setting, Url]) # insert the line into our article_i.tsv!

def get_scraping(input_path):
    path = str("./"+input_path)

    filenames = os.listdir(path)
    for i in range(1, 301):
        filenames = os.listdir(path + '/' + str(i))

        for file in filenames:
            with open(path + '/' + str(i) + './article_'+str(file.split("_")[1].replace(".html", ""))+'.tsv', 'w', encoding="utf-8", newline='') as out_file: # create for each html article its article_i.tsv according to the professor requests!
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerow(['bookTitle', 'bookSeries', 'bookAuthors', 'ratingValue', 
                                'ratingCount', 'reviewCount', 'Plot', 'NumberofPages', 'Published',
                                'Characters', 'Setting', 'Url'])
                scrap_book(tsv_writer, path + '/' + str(i) + "/" + file)

# creating the final dataset
def finaldataset_tsv(initial_path):
    path = str("./"+initial_path)
    suffix = ".tsv" # check the suffix
    filenames = os.listdir(path) # return a list that contain the entries of the folders present into the path passed
    data2 = pd.DataFrame()

    for i in range(1, 301):
        filenames = os.listdir(path + '/' + str(i))
        for file in filenames:
            if file.endswith(suffix): # check the .tsv suffix because there are .html extension
                with open(path + '/' + str(i) + '/article_'+str(file.split("_")[1]), 'r', encoding="utf-8", newline='') as out_file:
                        df = pd.read_csv(out_file,sep = "\t")
                        if  df.loc[0,"Plot"] != " " and df.loc[0,"bookTitle"] != " ": # check if the  article_i.tsv contains the main information, if it is insert in the final dataset
                            data2 = pd.concat([data2,df])
                            
    with open("finaloutput.tsv", "w", encoding="utf-8", newline="") as text_file: text_file.write(data2.to_csv(index=False)) # save the new file


################################ RQ2 ################################

def openCSV(filecsv):
    return pd.read_csv(filecsv, sep=",") # open the final dataset

# clean the text, create the bag of words
def clean_text(text):
    stop_words = set(stopwords.words('english')) # obtain the stop words
    good_words = [] # save the correct words to consider like tokens
    tokenizer = RegexpTokenizer("[\w']+") # function to recognize the tokens
    words = tokenizer.tokenize(text) # tokenize the text 
    for word in words:
        # check if the word is lower and it isn't a stop word or a number
        if word.lower() not in stop_words and word.isalpha(): 
            word = PorterStemmer().stem(word) # use the stemmer function
            good_words.append(word.lower()) # insert the good token to lower case
        
    return good_words

def create_vocabulary(df):
    ### Input == I use like input the dataset obtain in exercise 1 where i apply the clean text function
    ### Output == I obtain a vocabulary, the keys are all tokens (with no repeat) contained in the plot for the each rows
    ### for each token I define the index of the rows where the token is in the plot
    vocabulary = {}
    for i, row in df.iterrows():
            if len(df.at[i, "Plot"]) > 0:  # check if the list is empty or not to avoid the eventually error
                for word in df.at[i, "Plot"]: # bring the token from the list
                    if word in vocabulary.keys(): # insert the token into the vocabulary with the documents where this is present
                        if i not in vocabulary[word]:
                            vocabulary[word].append(i)
                    else:
                        vocabulary[word] = [i]
    return vocabulary

# create the inverted list according to the professors' requests
def create_inverted_list(vocabulary):
    inv_lst = {}
    indexes = list(vocabulary.keys()) # return the indexes list of the vocabulary
    for key in vocabulary.keys():
        term_id = indexes.index(key) # find the corresponding id from the vocabulary
        inv_lst[term_id] = vocabulary[key] # insert the list of documents into the inverted list
    
    return inv_lst

# map the interested word with corresponding term_id_i
def map_terms_id(vocabulary, cleanQString):
    # find each term_id
    term_id = []  # this is another function useful for mapping the term_id_i with the word into the vocabulary
    indexes = list(vocabulary.keys()) # return the indexes list of the vocabulary
    for token in cleanQString:
        term = indexes.index(token)
        term_id.append(term) # append the id that we want to make the score
        
    return term_id

# clean user's query
def cleanQuery(query):
    cleanQString = query.split(" ")
    
    stop_words = set(stopwords.words('english')) # obtain the stop words
    good_words = [] # save the correct words to consider like tokens
    tokenizer = RegexpTokenizer("[\w']+") # function to recognize the tokens
    
    for word in cleanQString:
        word = tokenizer.tokenize(word) # tokenize the text
        for w in word:
            if w.lower() not in stop_words and w.isalpha(): 
                w = PorterStemmer().stem(w) # use the stemmer function
                good_words.append(w.lower()) # insert the good token to lower case
    
    return good_words

# the search engine 1
def search_engine1(cleanQString, vocabulary, df, inv_lst):
    term_id = map_terms_id(vocabulary, cleanQString) # return the corresponding id of those terms

    # find the common documents where those terms are present
    intersection_list = []
    for term in term_id:
        if not intersection_list:
            intersection_list = inv_lst[term] # if the intersection list is empty insert the first list of the first token
        else:
            intersection_list = set(intersection_list).intersection(set(inv_lst[term])) # make the intersection, this respect the properties of the sets

    new_df = pd.DataFrame(columns=['bookTitle', 'Plot', 'Url']) # create the new dataset according to the professors' requests
    for row in intersection_list:
        #append row to the dataframe
        new_row = {'bookTitle': df.loc[row, "bookTitle"], 'Plot': df.loc[row, "Plot"], 'Url': df.loc[row, "Url"]}
        new_df = new_df.append(new_row, ignore_index=True)
        
    return new_df

##### RQ2.2

def inverted_list_2(vocabulary, df):
    ### Input == The vocabulary defined in function create_vocabulary
    ### Output == A new inverted list contained like keys all of token in the vocabulary but with the index and for each keys I have a list of tuples..
    ### The first value of tuple is the document where i can find this token and the second value is tfidf for the token in this document
    inv_lst2 = {}

    indexes = list(vocabulary.keys())
    for key in vocabulary.keys():
        lst_doc = vocabulary[key]

        result = []
        for doc in lst_doc:
            interested_row = df.at[doc, "Plot"] # extract the list of tokens from a proper column

            interested_word = key #i-th word

            tf = interested_row.count(interested_word) / len(interested_row)

            idf = math.log(len(df)/len(lst_doc))

            tf_idf = round(tf * idf, 3)

            result.append((doc,tf_idf))

        inv_lst2[indexes.index(key)] = result # insert the result into the inverted list

    return inv_lst2

def create_documents_list(df, inv_lst2, vocabulary):
    ### Input == The clean dataset, the new inverted list and the vocabulary
    ### Output == A list of dictionary defined documents. For each row in clean dataset i create a dictionary and in this dictionary
    ### I define like keys the token of Plot. For each token (keys) i use like values the tfidf.
    documents = []
    indexes = list(vocabulary.keys())
    for i, row in df.iterrows():
        tokens = {}
        for token in df.at[i, "Plot"]:
            tuple_list_values = inv_lst2[indexes.index(token)]
            for x in tuple_list_values:
                if x[0] == i:
                    tokens[token] = x[1]  
                    break

        documents.append(tokens)
    return documents

def similarity_score(df, inv_lst2, vocabulary, cleanQString, documents):
    ### Input == The clean dataset, the new inverted list, the query input but cleand, the vocabulary and the documents (list of dictionaries)
    ### Output == A list define top_k_documents where i define for each document the cosine similarity from the query cleaned and this document
    top_k_documents = []
    for i, row in df.iterrows():
        card_d_i = 1 / math.sqrt( sum(documents[i].values()) )
        somma = 0
        for token in cleanQString:
            try:
                somma += documents[i][token]
            except:
                somma += 0
        cosine_similarity = card_d_i * somma
        top_k_documents.append([round(cosine_similarity, 2),i])
        
    return top_k_documents

def heap_k_documents(top_k_documents, k):
    ### Input == the list top_k_documents
    ### Output == With heapq algotirhm i want to show the top 5 documents in ascending order respect the cosine similarity
    heapq.heapify(top_k_documents) 
    show_top_k_documents = (heapq.nlargest(k, top_k_documents)) 
    return show_top_k_documents

def dataset_search_engine2(show_top_k_documents, df):
    new_df = pd.DataFrame(columns=['bookTitle', 'Plot', 'Url', 'Similarity'])
    for row in show_top_k_documents:
        #append row to the dataframe
        new_row = {'bookTitle': df.loc[row[1], "bookTitle"], 'Plot': df.loc[row[1], "Plot"], 'Url': df.loc[row[1], "Url"], 'Similarity': row[0]}
        new_df = new_df.append(new_row, ignore_index=True)
    return new_df

################################ RQ3 ################################ (in this section we will use the functions defined above)

# cleaning the text column values, apply the function for each value through the corresponding column
def cleaning_value_columns(df1):
    df1['bookTitle'] = df1.bookTitle.apply(lambda x: clean_text(x))
    df1['bookAuthors'] = df1.bookAuthors.apply(lambda x: clean_text(x))
    df1['Plot'] = df1.Plot.apply(lambda x: clean_text(x))
    return df1
    
# create the new vocabulary
def create_newVocabulary(df1):
    vocabulary2 = {}
    for i, row in df1.iterrows():
        for column in ["bookTitle", "bookAuthors", "Plot"]: # insert the tokens into the new vocabulary
            if len(df1.at[i, column]) > 0:  # check if the list is empty or not to avoid the eventually error
                for word in df1.at[i, column]: # bring the token from the list
                    if word in vocabulary2.keys(): # insert the token into the vocabulary with the documents where this is present
                        if i not in vocabulary2[word]:
                            vocabulary2[word].append(i)
                    else:
                        vocabulary2[word] = [i]
    return vocabulary2

##### RQ3.2

# define the new inv_lst2 according to create the new score
def new_inv_lst2(vocabulary2, df1):
    inv_lst2 = {}

    indexes = list(vocabulary2.keys())
    for key in vocabulary2.keys():
        lst_doc = vocabulary2[key]

        result = []
        for doc in lst_doc:
            tf_idf = []
            for column in ["bookTitle", "bookAuthors", "Plot"]: # insert all tokens present in those columns
                interested_row = df1.at[doc, column] # extract the list of tokens from a proper column

                interested_word = key #i-th word

                # insert this construct because the interested_row could be empty, so put 0 like the term frequency
                try:
                    tf = interested_row.count(interested_word) / len(interested_row) 
                except:
                    tf = 0

                idf = math.log(len(df1)/len(lst_doc))

                tf_idf.append(round(tf * idf, 3))

            result.append((doc, round(sum(tf_idf)/3, 3))) # normalize the result

        inv_lst2[indexes.index(key)] = result # insert the result into the inverted list
        
    return inv_lst2

# define the documents of tokens with tf-idf for each token corresponding to the i-th document
def documents_list(vocabulary2, inv_lst2, df1):
    documents = [] 
    indexes = list(vocabulary2.keys()) # return the indexes list of the vocabulary
    for i, row in df1.iterrows():
        tokens = {} # insert the tokens and put its tf_idf score mapped in the i-th document
        for col in ["bookTitle", "bookAuthors", "Plot"]: # check those three columns
            for token in df1.at[i, col]:
                    tuple_list_values = inv_lst2[indexes.index(token)] # consider the list of documents

                    for x in tuple_list_values: # catch the documents where there is the tf_idf score of the token present into the document
                        if x[0] == i:
                            tokens[token] = x[1] # consider the score
                            break # break we find the interested term_id of the documents, we catch the score

        documents.append(tokens) # append the tokens with their tf_idf
        
    return documents

# new score and return top_k_documents according to the similarity formula and the boost considered by us
def newscore(df1, documents, cleanQString):
    top_k_documents = []
    for i, row in df1.iterrows():
        card_d_i = 1 / math.sqrt( sum(documents[i].values()) )

        somma = 0
        for token in cleanQString:
            try: # if the token isn't present the sum is equal to 0
                somma += documents[i][token]
            except:
                somma += 0

        cosine_similarity = card_d_i * somma

        top_k_documents.append([round(cosine_similarity, 2), i])  
     
    # boost it!
    for boost in top_k_documents:
        if float(df1.at[boost[1], "ratingValue"]) > 3.5 and int(df1.at[boost[1], "ratingCount"]) > 3500000:
            boost[0] += 2.0
        
    return top_k_documents

# create the dataset for this new search engine 2
def dataset_newsearch_engine2(show_top_k_documents, df):
    new_df = pd.DataFrame(columns=['bookTitle', 'Plot', 'Url', 'New-Score'])
    for row in show_top_k_documents:  #append row to the dataframe
        new_row = {'bookTitle': df.loc[row[1], "bookTitle"], 'Plot': df.loc[row[1], "Plot"], 'Url': df.loc[row[1], "Url"], 'New-Score': row[0]}
        new_df = new_df.append(new_row, ignore_index=True)
    
    return new_df

# sort and show the new_df according to the new score found
def newsearch_engine2(vocabulary2, df, df1, cleanQString, k):
    # create the new inverted list
    inv_lst2 = new_inv_lst2(vocabulary2, df1)
    
    # create the new document list
    documents = documents_list(vocabulary2, inv_lst2, df1)
    
    # obtain the new score so.. the top_k documents!
    top_k_documents = newscore(df1, documents, cleanQString)
    
    # sorted by score
    show_top_k_documents = heap_k_documents(top_k_documents, k)
    
    # create the dataset for this new search engine2
    new_df = dataset_newsearch_engine2(show_top_k_documents, df)
    
    return new_df

################################ RQ4 ################################

def newdata(df):
    ### input = dataset obtain in 1.3
    ### output = new dataset with some changes in bookSeries columns and Published columns
    
    newdf = df[df.bookSeries != ' '].reset_index(drop=True)
    newdf = df.dropna(subset = ["bookSeries"]).reset_index(drop=True)

    # clean the column value named "bookSeries"
    newdf["bookSeries"]=newdf["bookSeries"].str.replace("(", "").str.replace(")", "")
    newdf["bookSeries"]=newdf["bookSeries"].str.split("#") # split the text on #

    for i, rows in newdf.iterrows():
        if len(newdf.at[i,"bookSeries"]) < 2: # remove the bookSeries that have not the next books
            newdf = newdf.drop(index=i)

    newdf= newdf.reset_index(drop=True) # reset the indexes, because before we removed the bookSeries that have not the next books

    for i in range(len(newdf)):
        try:
            newdf.at[i,"Published"] = re.findall('([0-9]{4})', newdf.at[i,"Published"]) # if there is the date assigns in this cell the corresponding year
        except:
            pass

    return newdf

def new_dictionary(df):  
    ### input = dataset created from newdata function
    ### output = dictionary with 10 keys (the first 10 bookSeries obtain by dataset). Every keys contain a list of list
    ### every list contains the year passed from the first book of the series and cumulative pages
    
    TopSeries = []
    for i in range(0,10):
        TopSeries.append(df.at[i,"bookSeries"][0]) # appends the first 10 books to the TopSeries list

    dictionary=dict.fromkeys(TopSeries) #  create the dictionary using as keys the elements prenset into the TopSeries list

    for keys in dictionary.keys():
        dictionary[keys] = [] # create the list about the i-th token into the dictionary
        for i in range(len(df)):
            try:
                if len(df.at[i,"bookSeries"][1]) == 1 and df.at[i,"bookSeries"][0] == keys:
                    item = [int(df.at[i,"Published"][0]), int(df.at[i,"NumberofPages"])] # the item is composed by the year and the number of pages present into the book
                    dictionary[keys].append(item) # appends this item into the dictionary  
            except:
                pass

        dictionary[keys].sort(key = operator.itemgetter(0)) # for each key sort the list of values with respect to the year of the books

    for keys in dictionary:
        try:
            min_value = dictionary[keys][0][0] # take the min value of the list, the dictionary is sorted so we saved the first year
            count_y = 0
            for item in dictionary[keys]: # iterate the corresponding list and count the number of pages
                count_y += item[1]
                item[0] = (item[0] - min_value) # save the new difference of the year, considering the min_value
                item[1] = count_y # save the cumulative number of pages at each new iteration
        except:
            continue
    
    return dictionary

def plot_dictionary(dictionary):
    ### input == Take like input the dictionary create with new_dictionary function
    ### output == Create plot for each key in the dictionary. Each plot contain in x-axis 

    for keys in dictionary: # for each key present into the dictionary draw a plot
        x=[]
        y=[]

        for item in dictionary[keys]: # appends its values by x and y axes
            x.append(item[0])
            y.append(item[1])

        sns.set(font_scale=1)
        
        plt.style.use('seaborn-whitegrid')
        
        fig = plt.figure()
        
        ax = sns.barplot(x=x, y=y)
        
        ax.set_title(keys)
        
        ax.set(xlabel = 'Years from publication 1st book', ylabel = 'Cumulative series page',)

    return plt.show()