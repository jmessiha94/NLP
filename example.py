from word_vectorization import *
from pre_processing import pre_process

stop_words = ["i", "a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]


def get_test_data():
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
    proc_data = pre_process(X, tokenize=True, stop_words=stop_words)

    vec_X = np.array(bag_of_words(proc_data))
    
    return vec_X

	
def using_doc2vec(X):
    proc_data = pre_process(X, tokenize=True, stop_words=[])  
    
    model,vec_X = doc2vec_create_model(X,max_epochs=100,vec_size=10,alpha=0.025)
    
    model= Doc2Vec.load("Doc2Vec//apnews_dbow//doc2vec.bin")
    vec_X = doc2vec_use_model(proc_data,model)
    
    return vec_X


    
''' USING FUNCTIONS '''    
X,y = get_test_data()
bow = using_bag_of_words(X)
doc = using_doc2vec(X)