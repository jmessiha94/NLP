from word_vectorization import *		# for doc2vec and bag of words
from pre_processing import pre_process	# preprocessing text

## stop words, use whatever list you want to
stop_words = ["i", "a", "about", "above",...,"afterwards", "again", "against", "all"]

def get_test_data():
	# for testing purposes
    path = "sentiment_labelled_sentences//"
    
    # http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
    amazon = "amazon_cells_labelled.txt"
    imdb = "imdb_labelled.txt"
    yelp = "yelp_labelled.txt"

    files = [amazon,imdb,yelp]

    data = []
    labels = []
    
    # lets use all the files
    for file in files:
        f = open((path + file),'r')
        
    
        # read files
        for i in f.readlines():
            # split by tab
            i = i.split('\t')
            # add data
            data.append(str(i[0]))
            # add labels
            labels.append(int(i[1]))
    
    return data,labels


def using_bag_of_words(X):   
	# preprocess, words needs to be tokenized
    proc_data = pre_process(X, tokenize=True, stop_words=stop_words)

	# get BOW vector
    vec_X = bag_of_words(proc_data)
    
    return vec_X

	
def using_doc2vec(X):
	# preprocess
    proc_data = pre_process(X, tokenize=True, stop_words=[])  
    
	# no tokenize for doc2vec when creating model
    model,vec_X = doc2vec_create_model(X,max_epochs=100,vec_size=10,alpha=0.025)
    
	# example of loading in a model later
    model= Doc2Vec.load("Doc2Vec//apnews_dbow//doc2vec.bin")
	
	# tokenize when using model to convert other data
    vec_X = doc2vec_use_model(proc_data,model)
    
    return vec_X


    
''' USING FUNCTIONS '''    
X,y = get_test_data()
bow = using_bag_of_words(X)
doc = using_doc2vec(X)