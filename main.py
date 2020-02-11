
# for finding the number of threads available
import multiprocessing

# for stopwords
import nltk

# word2vec
import gensim.models.word2vec as w2v

# tsne
from sklearn.manifold import TSNE

# pca
from sklearn.decomposition import PCA

# used for array to hold vector representations of words
import numpy as np

# plotting
from matplotlib import pyplot

# reading in csv and creating a data frame
import pandas as pd

#for removing punctuation
import string

#used to time the transformations
import time

# frequent words
from collections import Counter

# for tokenizing
from nltk.tokenize import TweetTokenizer

# takes in the raw sentence and removes stopwords and punctuation. Ex.) ['Hello, is she here today?', ...] into [['hello','she','here','today'], ...]
def sentence_to_wordlist(raw):
    
    # remove nonprintable characters
    printable = set(string.printable)
    asciistring = ''.join(filter(lambda x: x in printable, raw))

    # create a set of stop words for removal.
    stops = set(nltk.corpus.stopwords.words('english'))

    # custom stop words -> since nltk stopword list is very limited. 
    custom_stopset = set(['dont','u','im','youre','ur','ill','thats','id','gonna','know','cant','shouldve','coulda','think','like','would','get','okay',
    'one','even','said','never','say','right','ever','make','let','ella','someone','much','got','actually','everyone','wanna','k','oh','tell'])
    
    # split sentence into words
    tokens = TweetTokenizer().tokenize(asciistring)

    # turn all tokens to lowercase
    tokens = [w.lower() for w in tokens]

    # remove stop words
    tokens = [w for w in tokens if not w in stops and w not in custom_stopset]

    # strip punctuation
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    
    # remove all non alphabet words
    words = [word for word in stripped if word.isalpha()]
    
    # return word list
    return words

# takes in the word2vec model, list of cb words, list of sexual words, list of both words, integer value to indicate skipgram was used
def plot_frequent_words(model,top_cb_words,top_sexual_words,top_both_words,sg):

        # used for title purposes when playing with CBOW and Skip Gram
        if sg == 0:
                pyplot.suptitle("cbow")
        else:
                pyplot.suptitle("sg")

        # create an empty vector array that will hold the 300 dimension vector representations for each of the words
        word_vector_arr = np.empty((0,300),dtype='f')
        
        # create a list to hold the actual words. used for plotting purposes
        word_labels = []

        # get number of elements in each list so that we can split word_vector_arr between cb words, sexual words, and both words
        cb_list_size = len(top_cb_words)
        sexual_list_size = len(top_sexual_words)

        #The order of which we will append the word vector arr will be cb words -> sexual words -> both words. We will use
        # end_of_sexual_words for array slicing purposes when plotting
        end_of_sexual_words = cb_list_size + sexual_list_size
        
        # we grab the vector representation of each word in cb words and append to arr. and append the actual word to be a label later
        for wrd in top_cb_words:

                # __getitem__ grabs the vector representation of 'wrd'
                cb_wrd_vector = model.wv.__getitem__(wrd)
                
                #append the actual word to word labels
                word_labels.append(wrd)
                
                #append the vector to the word vector array
                word_vector_arr = np.append(word_vector_arr,np.array([cb_wrd_vector]),axis=0)

        # we grab the vector representation of each word in sexual words and append to arr. and append the actual word to be a label later
        for wrd in top_sexual_words:

                # __getitem__ grabs the vector representation of 'wrd'
                sexual_wrd_vector = model.wv.__getitem__(wrd)
                
                #append the actual word to word labels
                word_labels.append(wrd)
                
                #append the vector to the word vector array
                word_vector_arr = np.append(word_vector_arr,np.array([sexual_wrd_vector]),axis=0)


        # we grab the vector representation of each word in both words and append to arr. and append the actual word to be a label later
        for wrd in top_both_words:
                
                # __getitem__ grabs the vector representation of 'wrd'
                both_wrd_vector = model.wv.__getitem__(wrd)
                
                #append the actual word to word labels
                word_labels.append(wrd)
                
                #append the vector to the word vector array
                word_vector_arr = np.append(word_vector_arr,np.array([both_wrd_vector]),axis=0)


        #######################################
        # start t-SNE plot
        #######################################  

        # t-SNE plot located at row 1 column 1 in a 1x2 subplot
        pyplot.subplot(1,2,1) 

        #initialize tsne 2 dimensions, and random state is virtually the 'seed' for consistent results
        tsne = TSNE(n_components=2,random_state=0)
        
        # using a fixed print notation. else scientific notation will be used
        np.set_printoptions(suppress=True)
        
        # time tsne transformation
        start_time = time.time()
        Y_tsne = tsne.fit_transform(word_vector_arr)
        print("Time for TSNE transformation: " + str(time.time()-start_time)) 

        # optional print of X and Y coordinates 
        #print(Y_tsne)

        # extracting x coordinates and y coordinates. Format of Y_tsne -> [(x1,y1),(x2,y2), ...]
        x_coords_tsne = Y_tsne[:,0]
        y_coords_tsne = Y_tsne[:,1]


        # specify scatterplot - split array between cb words (in red), sexual words (in blue), and both words (in cyan)
        tsnered = pyplot.scatter(x_coords_tsne[:cb_list_size],y_coords_tsne[:cb_list_size],marker='o',c='red')
        tsneblue = pyplot.scatter(x_coords_tsne[cb_list_size:end_of_sexual_words],y_coords_tsne[cb_list_size:end_of_sexual_words],marker='o',c='blue')
        tsnecyan = pyplot.scatter(x_coords_tsne[end_of_sexual_words:],y_coords_tsne[end_of_sexual_words:],marker='o',c='c')

        # labels the data at each coordinate pair
        for label, x, y in zip(word_labels,x_coords_tsne,y_coords_tsne):
                pyplot.annotate(label,xy=(x,y),xytext=(0,0),textcoords='offset points')
        
        # title for t-sne plot
        title = "(TSNE) most frequent words: "
        
        # adds a legend
        pyplot.legend((tsnered,tsneblue,tsnecyan),('Cyberbullying','Sexual','Both'),loc='lower left',fontsize='8')
        
        # adds title
        pyplot.title(title) 

        #######################################
        # start PCA plot
        #######################################                
        
        #pca plot located at row 1 column 2 plot in a 1x2 plot
        pyplot.subplot(1,2,2)

        #initialize pca, 2 dimensions, random_state is virtually the 'seed' for consisitent results
        pca = PCA(n_components=2,random_state=0)

        
        # using a fixed print notation. else scientific notation will be used
        np.set_printoptions(suppress=True) 
        
        #time pca transformation
        start_time = time.time()
        Y_pca = pca.fit_transform(word_vector_arr)
        print("Time for PCA transformation: " + str(time.time()-start_time))

        # optional print of X and Y coordinates
        #print(Y_pca)
        
        # extracting x coordinates and y coordinates. Format of Y_pca -> [(x1,y1),(x2,y2), ...]
        x_coords_pca = Y_pca[:,0]
        y_coords_pca = Y_pca[:,1]

        #specify scatterplot - split array between cb words (in red), sexual words (in blue), and both words (in cyan)
        pcared = pyplot.scatter(x_coords_pca[:cb_list_size],y_coords_pca[:cb_list_size],marker='o',c="red")
        pcablue = pyplot.scatter(x_coords_pca[cb_list_size:end_of_sexual_words],y_coords_pca[cb_list_size:end_of_sexual_words],marker='o',c='blue')
        pcacyan = pyplot.scatter(x_coords_pca[end_of_sexual_words:],y_coords_pca[end_of_sexual_words:],marker='o',c='c')
        
        # add labels to each coordinate pair
        for label, x, y in zip(word_labels,x_coords_pca,y_coords_pca):
                pyplot.annotate(label,xy=(x,y),xytext=(0,0),textcoords='offset points')

        #title for pca plot
        title = "(PCA) most frequent words: "

        # add legend to plot
        pyplot.legend((pcared,pcablue,pcacyan),('Cyberbullying','Sexual','Both'),loc='lower left',fontsize='8')
        
        # add title
        pyplot.title(title)
        
        # show the plot
        pyplot.show()


# this will take in sentence wordlist with the format [['hello','she','here','today'], ...] and turns it into, ['hello','she','here','today', ...]
# pretty much just getting rid of the sublists
def allWords(sentences):
        wordlist = []
        for sentence in sentences:
                for word in sentence:
                        wordlist.append(word)
        
        return wordlist


# read in csv files. df_main commented out since we will be loading in the pretrained model for consistent results
df_main = pd.read_csv('allData.csv')
df_cb = pd.read_csv('cb_out.csv')
df_sexual = pd.read_csv('sexual_out.csv')

# gather all answers and questions in their respective lists. again df main commented out.
#main_answer_sentences = df_main['answer'].values.tolist()
#main_question_sentences = df_main['question'].values.tolist()

cb_answer_sentences = df_cb['answer'].values.tolist()
cb_question_sentences = df_cb['question'].values.tolist()

sexual_answer_sentences = df_sexual['answer'].values.tolist()
sexual_question_sentences = df_sexual['question'].values.tolist()

raw_cb_sentences = cb_answer_sentences+cb_question_sentences
raw_sexual_sentences = sexual_answer_sentences+sexual_question_sentences
#raw_main_sentences = main_answer_sentences+main_question_sentences


# append the words after preprocessing sentences into wordlist
cb_wordlist = []
sexual_wordlist = []
#main_wordlist = []


# create two seperate wordlist. one for cyberbullying and the other for sexual. These wordlist will be later used to extract frequent words
print('creating cb wordlist')
for raw_sentence in raw_cb_sentences:
        cb_wordlist.append(sentence_to_wordlist(raw_sentence))

print('creating sexual wordlist')
for raw_sentence in raw_sexual_sentences:
        sexual_wordlist.append(sentence_to_wordlist(raw_sentence))

###########################################################################################
###### Section used to create the wordlist that will be used to train the word2vec model
###########################################################################################
'''
errorCount = 0
print('creating main wordlist')
for raw_sentence in raw_main_sentences:
        try:
                main_wordlist.append(sentence_to_wordlist(raw_sentence))
        except:
                errorCount+=1

print(str(errorCount) + ' failed sentence cleanings occurs when creating main wordlist')
'''
###########################################################################################


# turn wordlist into a list of all words
print('extracting most frequent words in each category')
cb_words = allWords(cb_wordlist)
sexual_words = allWords(sexual_wordlist)

# count how often each word occurs
Counter_cb = Counter(cb_words)
Counter_sexual = Counter(sexual_words)

# grab the top 30 words in both categories. these two lists will have the format, [['word','12'],...], 
# where 'word' is some word that is frequently used, and the number shows how often it is used
most_frequent_cb_words = Counter_cb.most_common(30)
most_frequent_sexual_words = Counter_sexual.most_common(30)


# raw sets will be used to add the words within the "most_frequent_words" lists above and later used to 
# get words that occur in both sets, and words that occur only in each of the sets
raw_cb_set = set()
raw_sexual_set = set()

# these sets will be used to extract words that happen in both categories, and words that happen in seperate categories
both_set = set()
only_cb_set = set()
only_sexual_set = set()

# raw sets being used to add the words from the most_frequent_words list
for cb_word in most_frequent_cb_words:
        raw_cb_set.add(cb_word[0])

for sexual_word in most_frequent_sexual_words:
        raw_sexual_set.add(sexual_word[0])

# extracting words that occur in both, and words that only occur in one category
both_set = raw_cb_set & raw_sexual_set
only_cb_set = raw_cb_set.difference(raw_sexual_set)
only_sexual_set = raw_sexual_set.difference(raw_cb_set)

# convert the sets to lists
both_list = list(both_set)
only_cb_list = list(only_cb_set)
only_sexual_list = list(only_sexual_set)

# Show the words in all lists. Just for vizualization purposes
print(both_list)
print(only_cb_list)
print(only_sexual_list)

#########################################################################################################
####### Section used to build the word2vec model and train the main_wordlist that is commented out above
#########################################################################################################
'''
## predefine the word2vec model parameters
# number of dimensions for each word vector
num_features = 300

# minimum word count threshold (word must occur at least 3 times to be considered by the model)
min_word_count = 1

# using max threads to train the model
num_workers = multiprocessing.cpu_count()

# keep consistent results
seed = 1

# skipgram? if yes 1 if no 0 (if no we will be using continuous bag of words (CBOW))
sg = 1

# creating word2vec model with prespecified parameters
print("building word2vec model")
model = w2v.Word2Vec(sg=sg,seed=seed,workers=num_workers,size=num_features,min_count=min_word_count)

# build the models vocabulary to be later used for training
print('building model vocab...')
model.build_vocab(main_wordlist)

# number of words in models vocabulary
print('model vocab list: ' + str(len(model.wv.vocab)))

# train the word2vec model 
print('training model...')
model.train(sentences=main_wordlist,total_examples=model.corpus_count,epochs=model.epochs) '''
#################################################################################################################



# optional save and load. 
#model.save("model_cbow")
print("loading model....")
model = w2v.Word2Vec.load("model_sg")

print("plotting...")
plot_frequent_words(model,only_cb_list,only_sexual_list,both_list,1)
