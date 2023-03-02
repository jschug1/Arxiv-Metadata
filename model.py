# This file contains a classifying NN model which categorizes papers into subject based on title.
# This is not meant to be ground break, just a pet project.
# The categories we use just arise from the arxiv taxonomy system, and are computer science, economics,
# electrical engineering, math, physics, quantitative biology, quantitative finance, and statistics.  The data is skewed
# towards CS, math, and physics since those subjects have the most papers by far.
# I am aware that the built in tokenizer in tf would have been helpful, but I decided to do this from scratch,
# to practice.

########################################################################################
# First, we generate the training and validation data.
# For the purposes of this project, I hand picked 5000 papers from each topic.
# The 5000 number came about since some topics (i.e. econ, qfin, etc)
# have only about 5000 papers in the dataset to begin with.
# To even it out, I restricted to 5000 in the more popular subjects as well.
# Then 4000 of those will go into the training data and 1000 will be for validation.
# For simplicity, we lump in the various subfields of physics into one label, physics.
########################################################################################

# dictionaries tokenizing the categories to integers.
# the first two dictionaries take the categories from the arxiv metadata and associate them to integers
# the last one is for the prediction output.
category_dict = {0: ['cs'], 1: ['econ'], 2: ['eess'], 3: ['math'],
                 4: ['astro-ph', 'cond-mat', 'gr-qc', 'hep', 'nlin', 'nucl', 'physics', 'quant'],
                 5: ['q-bio'], 6: ['qfin'], 7: ['stat']}
reverse_category_dict = {'cs.': 0, 'eco': 1, 'ees': 2, 'mat': 3,
                 'ast': 4, 'con': 4, 'gr-': 4, 'hep': 4, 'nli': 4, 'nuc': 4, 'phy': 4, 'qua': 4,
                 'q-b': 5, 'q-f': 6, 'sta': 7}
topic_dict = {0: 'cs', 1: 'econ', 2: 'eess', 3: 'math',
                 4: 'physics',
                 5: 'q-bio', 6: 'qfin', 7: 'stat'}

########################################################################################
# This function isolates the 'category' value from a paper dictionary in 'arxiv-metadata' and isolates the first
# classifier listed in the arxiv taxonomy.  Papers with multiple listed categories will take the first one listed.
#
# Input: paper = a dictionary of paper metadata from 'arxiv-metadata'
#
# Output: label = an integer 0-7 corresponding to the subject of the paper, see dictionaries above.
########################################################################################

def find_category(paper):

    category = paper['categories']
    category = category.split(' ')
    label = 9
    for subcat in category:
        initial = subcat[:3]
        if initial in reverse_category_dict.keys():
            label = reverse_category_dict[initial]
            break

    return label

########################################################################################
# word_list: takes a string of text, geared toward academic writing, and makes a list of words from the text
#               distinct from split() in that it does a bit of preprocessing, i.e. removes extra spaces and math symbols
#               Currently does not take into account capitalizations from beginnings of sentences
#
# input: str = a string of text
#
# output: words = a list of words which occur in the text minus math symbols or weird spacing issues
########################################################################################

def word_list(str):
    str = remove_latex_eqns(str)
    chars_to_remove = ['(', ')', '.', ',']

    str = str.replace('\n', ' ')

    for char in chars_to_remove:
        str = str.replace(char, '')

    str = str.split(' ')

    for i in range(len(str)):
        str[i] = str[i].lstrip()
        str[i] = str[i].rstrip()
        str[i] = str[i].lower()


    words = []
    for word in str:
        if word != '':
            words.append(word)

    return words

########################################################################################
# Now we can define a function which creates the data set with desired parameters listed above.
########################################################################################

def make_datasets():
    type_num = [0, 0, 0, 0, 0, 0, 0, 0] # keeps track of the number of papers of each topic in the loop
    training_papers = []
    validation_papers = []

    # Now 'arxiv-metadata' has about 2 million papers, but want to isolate 40000 of them divided evenly among the
    # topics, this for loop achieves that
    index = 0
    for entry in open('arxiv-metadata.json', 'r'):
        if index % 100000 == 0:
            print(f'index = {index}') # for purposes of tracking progress
        paper = json.loads(entry)
        topic = find_category(paper) # calls the find_category function
        type_num[topic] += 1
        if type_num[topic] <= 4000: # puts the first 4000 of a subject into the training data
            training_papers.append(paper)
        if type_num[topic] > 4000 and type_num[topic] <= 5000: # takes the next 1000 and puts them into validation
            validation_papers.append(paper)
        index += 1

    all_papers = training_papers + validation_papers # all 40000 papers

    # the following loop generates the tokens for each distinct word in the titles of the data set.
    # references the word_list function

    title_words = []
    for paper in all_papers:
        title = paper['title']
        title = word_list(title)
        title_words += title
    title_words = list(set(title_words)) # keeps only distinct words
    token_list = []
    i = 0
    for word in title_words:
        token_list.append([i, word])
        i += 1

    token_dict = {i: word for i, word in token_list} # creates integer tokens from distinct words
    reverse_dict = {word: i for i, word in token_list} # can recover token from word

    # The following loop then makes np arrays for the data.  It takes the title, turns into a list of integers from a
    # token, then adds it as a row in an array.

    training_data = []
    training_labels = []
    validation_data = []
    validation_labels = []
    for paper in training_papers:
        title = word_list(paper['title'])
        tokened_title = []
        for word in title:
            tokened_title.append(reverse_dict[word])
        training_data.append(tokened_title)
        training_labels.append(find_category(paper))
    for paper in validation_papers:
        title = word_list(paper['title'])
        tokened_title = []
        for word in title:
            tokened_title.append(reverse_dict[word])
        validation_data.append(tokened_title)
        validation_labels.append(find_category(paper))

    all_data = training_data + validation_data
    all_labels = training_labels + validation_labels

    return training_data, training_labels, validation_data, validation_labels, token_dict, reverse_dict

########################################################################################
# Now to instantiate the make data function
########################################################################################

training_data, training_labels, validation_data, validation_labels, token_dict, reverse_dict = make_datasets()
all_data = training_data + validation_data

########################################################################################
# max_len: finds the max length title from the data
#
# input: data = a list of lists
#
# output: max = integer max length list from data
########################################################################################

def max_len(data):
    max = 0
    for title in all_data:
        if len(title) > max:
            max = len(title)
    return max

########################################################################################
# pad_sequences: manually pads the lists to be all the same length, which is the length of the longest one
#                   also converts data into np arrays.  Again I am well aware that tf does this, but I wanted to
#                   do it manually to practice.
#
# input: data = a list of lists
#
# output: same data but with 0's padded at the end to make every list the same length and turned into an np array
########################################################################################

def pad_sequences(train_data, valid_data):

    data = train_data + valid_data
    maxlen = max_len(data)

    for title in train_data:
        while len(title) <= maxlen:
            title.append(0)

    for title in valid_data:
        while len(title) <= maxlen:
            title.append(0)

    for i in range(len(train_data)):
        train_data[i] = np.array(train_data[i])
    for i in range(len(valid_data)):
        valid_data[i] = np.array(valid_data[i])

    train_data = np.array(train_data)
    valid_data = np.array(valid_data)

    return train_data, valid_data, maxlen

########################################################################################
# Now we can pad the training and validation data sequences as well as convert the labels into numpy arrays.
########################################################################################

training_data, validation_data, maxlen = pad_sequences(training_data, validation_data)

training_labels = np.array(training_labels)
validation_labels = np.array(validation_labels)

########################################################################################
# Now we can create the model.
########################################################################################

import random

import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

########################################################################################
# We define a few constants which we will use in the make_model function
########################################################################################

buffer_size = 10000
batch_size = 32
embedding_dim = 64
vocab_size = len(token_dict)
epochs = 5

########################################################################################
# Now we can make the model.
########################################################################################


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=maxlen+1),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

########################################################################################
# This will train the model and record results to the various history.
########################################################################################

history = model.fit(
    x=training_data,
    y=training_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(validation_data, validation_labels),
)

########################################################################################
# This plots the accuracy and loss of the training and validation data over the epochs.
########################################################################################

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

########################################################################################
# Now we can make the function which predicts the topic of a paper based on the title.  Note that out of token words
# are assigned token 0.
########################################################################################

def title_to_topic(model, str):
    temp = word_list(str)
    ttemp = []
    for word in temp:
        if word in reverse_dict.keys():
            ttemp.append(reverse_dict[word])
        else:
            ttemp.append(0)
    while len(ttemp) <= maxlen:
        ttemp.append(0)
    ttemp = np.array(ttemp)
    title = np.transpose(tf.expand_dims(ttemp, axis=-1))
    classes = model.predict(title, batch_size=10, verbose=0)
    classes = classes[0]
    topic = topic_converter(np.argmax(classes)) # predicts based on max probability
    percentage = classes[np.argmax(classes)]*100

    return topic, percentage

def topic_converter(integer):
    category_dict = {1: 'cs', 2: 'econ', 3: 'eess', 4: 'math',
                     5: 'physics',
                     6: 'q-bio', 7: 'qfin', 8: 'stat'}
    return topic_dict[integer]

########################################################################################
# Give it a shot! It will also print the probability from the softmax layer at the end.
########################################################################################

topic = title_to_topic(model, "Confluence of the K-theoretic I-function for Calabi-Yau manifolds and an Application to the Integrality of the Mirror Map")
print(topic)



