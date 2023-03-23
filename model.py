# This file contains a classifying NN model which categorizes papers into subject based on title.
# This is not meant to be ground break, just a pet project.
# The categories we use just arise from the arxiv taxonomy system, and are computer science, economics,
# electrical engineering, math, physics, quantitative biology, quantitative finance, and statistics.  The data is skewed
# towards CS, math, and physics since those subjects have the most papers by far.
# I am aware that the built-in tokenizer in tf would have been helpful, but I decided to do this from scratch,
# to practice.

import json
import pandas as pd
import random
import tensorflow as tf
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt

########################################################################################################################
# First, write the functions to classify the papers relative to a dictionary which associates
# arxiv categories to integer labels the model will use.  This document will generate data and labels relative to
# the dictionary you choose.
#
# Additionally, cross_topic = True means that the labels will track multiple categories, while
# cross_topic = False will only use the main topic for the label.
########################################################################################################################

# associates the all arxiv labels to integers separately
# category_dict = {'cs.': 0, 'eco': 1, 'ees': 2, 'mat': 3,
# 'ast': 4, 'con': 5, 'gr-': 6,
# 'hep': 7, 'nli': 8, 'nuc': 9, 'phy': 10, 'qua': 11,
# 'q-b': 12, 'q-f': 13, 'sta': 14}

# example dictionary for math or physics
math_or_phys_dict = {'mat': 0, 'ast': 1, 'con': 1, 'gr-': 1,
                     'hep': 1, 'nli': 1, 'nuc': 1, 'phy': 1, 'qua': 1}

# simplifies to math or not
# math_dict = {'mat': 0}

# For this example we will use the math_dict and since it is simple we do not want to track multiple category papers
cat_dictionary = math_or_phys_dict
cross_topic = True


########################################################################################################################
# The following functions help generate labels for the cross_topic = True case.
########################################################################################################################

# this function takes a list of subsets of a list
def generate_subsets(temp_list):
    if not temp_list:
        return []

    if len(temp_list) == 1:
        return [[], temp_list]

    list1 = generate_subsets(temp_list[1:])
    list2 = copy.deepcopy(list1)
    for i in range(len(list1)):
        list2[i].insert(0, temp_list[0])

    subset_list = list1 + list2

    return subset_list


# Generates all the labels for multiple categories given a dictionary, cat_dict.
# Also generates a numbering for these labels for the sake of the NN stored as a dictionary
# For example, math_or_phsy_dict, cross_topic=True gives labels
# (2,) -> neither math nor physics -> 0
# (1,) -> only physics -> 1
# (0,) -> only math -> 2
# (0,1) -> math and physics -> 3
def generate_list_labels(cat_dict, cross):
    max_val = max(cat_dict.values())

    ints_list = list(range(max_val + 1))
    label_lists = generate_subsets(ints_list)

    label_lists[0].append(max_val + 1)

    for ind in range(len(label_lists)):
        label_lists[ind] = [tuple(label_lists[ind]), ind]

    label_dict = {tup: ind for [tup, ind] in label_lists}

    if cross:
        return label_dict

    else:
        len_one_keys = []
        for key in label_dict.keys():
            if len(key) == 1:
                len_one_keys.append(key)
        exclusive_label_dict = {key: key[0] for key in len_one_keys}
        return exclusive_label_dict


########################################################################################################################
# The 'category' function generates the label of a paper given a string of arxiv categories
# relative to your desired cat_dict and cross_topic = Bool value.
########################################################################################################################


def category(text, cat_dict, cross):
    max_val = max(cat_dict.values())

    # initializes the label list
    label_list = []

    # splits categories by spaces
    text = text.split(' ')

    # iterate over the list of categories
    for temp_topic in text:

        # take only first 3 letters to turn arxiv labels into integers
        inits = temp_topic[:3]

        # turn category into integer by the cat_dict dictionary
        if inits in cat_dict.keys():
            label_list.append(cat_dict[inits])

    # only take distinct integers and sort for convenience

    if cross == True:
        if label_list == []:
            return (max_val + 1,)
        else:
            label_list = sorted(list(set(label_list)))
            category_tup = tuple(label_list)
            return category_tup

    else:
        if label_list == []:
            return (max_val + 1,)
        else:
            return (label_list[0],)


########################################################################################################################
# Now we write the functions to identify the important words in the titles to help classify them by topic.
# Not entirely necessary, but I am hoping that identifying only the most specific words to a category, the data
# will be less noisy for the NN.
########################################################################################################################

# list of words to not include in the title words
stop_words = ['and', 'with', 'the', 'a', 'of', 'in', 'for', 'on', 'to', 'from', 'an', 'at', 'by', 'using']


# remove latex eqns from a string, a bit naive and only detects dollar signs or \[ \] notation.
# an attempt to remove as many generic variable names as possible
def remove_latex_eqns(test_str):
    dolla = -1
    eqn = 0
    newstr = ''
    for letter in test_str:
        if letter == '$':
            dolla = -1 * dolla
            continue
        if letter == '[' and dolla == -1:
            eqn += 1
            continue
        if dolla == -1:
            if eqn == 0:
                newstr += letter
        if letter == ']':
            eqn = eqn - 1
            continue
        # print(f'letter = {letter}')
        # print(f'dolla = {dolla}')
        # print(f'eqn = {eqn}')

    return newstr


# get list of important words to an academic paper title from a string
def word_list(test_str, temp_stop_words):
    # removes latex eqns to isolate words
    test_str = remove_latex_eqns(test_str)

    # removes next line character
    test_str = test_str.replace('\n', ' ')

    # removes dashes
    test_str = test_str.replace('-', ' ')

    # list of punctuation to ignore
    chars_to_remove = ['(', ')', '.', ',', ':']

    # removes all the punctuation from chars_to_remove
    for char in chars_to_remove:
        test_str = test_str.replace(char, '')

    # splits by spaces into a list of words
    test_str = test_str.split(' ')

    # takes every word and removes spaces from beginning to end and makes it lowercase
    for ind in range(len(test_str)):
        test_str[ind] = test_str[ind].lstrip()
        test_str[ind] = test_str[ind].rstrip()
        test_str[ind] = test_str[ind].lower()

    # takes previous list and adds non-stop words into a new list
    words = []
    for temp_word in test_str:
        if temp_word != '':
            if temp_word not in temp_stop_words:
                words.append(temp_word)

    return words


# generates a dictionary with all the word appearance frequency data
# keys = all words mentioned in titles
# value = a list of the integers counting the number of times that word appeared in a title with that label
# for example, math_or_phys_dict will have a list of 4 values, a count in each of the 4 possible labels
def make_word_distr(cat_dict, cross):
    list_labels = generate_list_labels(cat_dict, cross)  # calls generate_list_labels
    number_of_labels = len(list_labels)  # number of labels

    temp_word_distr_dict = {}  # initializes the dictionary
    temp_totals_list = [0] * number_of_labels  # initializes a total count of papers with label

    ind = 0
    for temp_entry in open('arxiv-metadata.json', 'r'):

        value = float(ind / 2207558) * 100  # for purposes of progress tracking
        print(f'making distr freq {value:.2f} %', end='\r')
        sys.stdout.flush()

        temp_paper = json.loads(temp_entry)  # loads paper

        temp_title = temp_paper['title']  # loads title
        temp_title = word_list(temp_title, stop_words)  # gets list of words from title by word_list

        categories = temp_paper['categories']  # gets string of arxiv categories
        topics = category(categories, cat_dict, cross)  # turns into tuple using cat_dict
        label = list_labels[topics]  # turns into label by list_of_labels
        temp_totals_list[label] += 1  # adds to count

        # adds data to dictionary from list of words
        for temp_word in temp_title:
            if temp_word in temp_word_distr_dict.keys():
                temp_word_distr_dict[temp_word][label] += 1
            else:
                temp_word_distr_dict[temp_word] = [0] * number_of_labels
                temp_word_distr_dict[temp_word][label] += 1

        ind += 1

    return temp_word_distr_dict, temp_totals_list, list_labels


# instantiates the above function
word_distr_dict, totals_list, list_of_labels = make_word_distr(cat_dictionary, cross=cross_topic)

# converts to pandas dataframe for simplicity
df = pd.DataFrame(word_distr_dict)
df = df.transpose()

print(df)
print(totals_list)
print(list_of_labels)


########################################################################################################################
# Now we can process the dataframe to try and isolate words with the most information about topic the paper is about
# We can do this by defining some kind of metric of how much each word is specific to a label.
# For this document, the outlier_metric function tries to be such a metric.
# Given a count [# of appearances in label 0, # of appearances in label 1, etc...],
# the metric takes the max of (ith entry - average of j != i entries). The bigger this max,
# the most one count stands out, and the more that word is specific to one label.
# To compensate for the fact that there may be more papers in one category than another, this is done relative
# to the total amount of papers.  So actually it is compute on % of appearances in label, rather than total number.
# This is not entirely useful if we are tracking cross category papers, since they may have words from both fields.
########################################################################################################################

# adds numbers in list
def sum_over(num_list):
    temp_sum = 0
    for num in num_list:
        temp_sum += num
    return temp_sum


# takes average of numbers in list
def average(num_list):
    temp_sum = sum_over(num_list)
    length = len(num_list)

    avg = float(temp_sum) / float(length)

    return avg


# computes the metric described above
def outlier_metric(num_list):
    dist_to_avg_without = [0] * len(num_list)

    for n in range(len(num_list)):
        omit_list = num_list[:n] + num_list[n + 1:]
        avg = average(omit_list)
        dist = abs(num_list[n] - avg)
        dist_to_avg_without[n] = dist

    max_dist = max(dist_to_avg_without)

    return max_dist


# adds a column with the metric value for each word in the data frame
# we can either let totals_list = paper_year to do this relative to the total amount of papers with that category,
# or we can let paper_year = [1,1,...,1] if we want to measure outliers from total count
def add_measure_col(frame, measure, paper_year, string):
    frame[string] = 0

    for ind, row in frame.iterrows():

        num_list = []
        for n in range(len(list_of_labels)):
            num_list.append(row[n])
        for n in range(len(list_of_labels)):
            num_list[n] = float(num_list[n]) / float(paper_year[n])
        distance = measure(num_list)
        frame.loc[ind, string] = distance

    return frame


# adds column of dataframe for the outlier metric
df = add_measure_col(df, outlier_metric, totals_list, 'outlier metric')
print(df)


########################################################################################################################
# This function weeds out words with low outlier metrics, meaning they are not sufficiently specific to any one field
# and thus are potentially noise for the NN.
# The variable cutoff can be adjusted for how many words you want to throw out.  cutoff = 0 means not throwing out any.
# This is advisable in the situation where you have cross category labels, since field specific
# buzzwords may be common to more than one label.
# We can also use the words with sufficiently high metric to make the tokenizer.
########################################################################################################################

# makes small dataframe of only words with metric value above cutoff value
def weed_out(frame, column_name, temp_cutoff):
    drop_names = []

    if temp_cutoff == 0:
        return frame

    for ind, row in frame.iterrows():

        if frame.loc[ind, column_name] < temp_cutoff:
            drop_names.append(ind)

    new_frame = frame.drop(index=drop_names)

    return new_frame


# Cutoff is set to 0 since we are tracking multiple category labels, important words may not be too specific to labels.
cutoff = 0

df_mini = weed_out(df, 'outlier metric', cutoff)
print(df_mini)

# makes a list of all the words in df_mini.  Then we can check only these words in the titles of papers when
# generating the data
keywords = list(df_mini.index.values)
print(len(keywords))

# turns list of keywords into tokenizer dictionaries. not that the count starts at 1 so that 0 is the OOV token
for i in range(len(keywords)):
    keywords[i] = [keywords[i], i]

token_dict = {i + 1: word for [word, i] in keywords}  # creates integer tokens from distinct words
reverse_dict = {word: i + 1 for [word, i] in keywords}  # recover token from word


########################################################################################################################
# Now we can generate the data using the titles, the keywords, and the tokenizer.
########################################################################################################################

# splits the data set into training and validation
def split_list(test_list, split_percent):
    split_num = int(len(test_list) * split_percent)
    list1 = test_list[:split_num]
    list2 = test_list[split_num:]

    return list1, list2


# improved word list function, which now only includes words from the tokenizer dictionary of keywords
# admittedly not the greatest implementation of this function since I don't need stop words anymore
def improved_word_list(test_str, temp_stop_words, frame, metric, temp_cutoff):
    words = word_list(test_str, temp_stop_words)
    new_words = []
    for tmep_word in words:
        measure = frame.loc[tmep_word, metric]
        if measure > temp_cutoff:
            new_words.append(tmep_word)

    return new_words


# parameter dictating the amount of titles with a certain label appear in the data set
data_size = 5000


# makes the data set, outputs training data and labels, as well as validation data and labels as np arrays
def make_datasets(cat_dict, cross, data_size, split_num, cutoff_num):
    list_labels = generate_list_labels(cat_dict, cross)
    number_of_labels = len(list_labels)

    type_num = [0] * number_of_labels
    all_papers = []

    ind = 0
    for temp_entry in open('arxiv-metadata.json', 'r'):

        value = float(ind / 2207558) * 100
        print(f'making data set, {value:.2f} %', end='\r')
        sys.stdout.flush()

        temp_paper = json.loads(temp_entry)

        temp_title = temp_paper['title']
        topics = category(temp_paper['categories'], cat_dict, cross=True)  # calls the find_category function
        label = list_labels[topics]
        type_num[label] += 1

        words = improved_word_list(temp_title, stop_words, df, 'outlier metric', cutoff_num)

        if type_num[label] <= data_size:
            all_papers.append([words, label])

        ind += 1

    for pair in all_papers:
        temp_title = pair[0]
        num_title = []
        for temp_word in temp_title:
            if temp_word in reverse_dict.keys():
                num_title.append(reverse_dict[temp_word])
        pair[0] = num_title

    random.shuffle(all_papers)
    titles = []
    labels = []

    for pair in all_papers:
        titles.append(pair[0])
        labels.append(pair[1])

    train_data, valid_data = split_list(titles, split_num)
    train_labels, valid_labels = split_list(labels, split_num)

    return train_data, train_labels, valid_data, valid_labels


# instantiates the make_datasets function
training_data, training_labels, validation_data, validation_labels = make_datasets(cat_dictionary,
                                                                                   cross=cross_topic,
                                                                                   data_size=data_size,
                                                                                   split_num=0.9,
                                                                                   cutoff_num=cutoff)

# recombines all the data for the padding step
all_data = training_data + validation_data


########################################################################################################################
# Now we pad the sequences of tokens to normalize the data.  manually pads the lists to be all the same length,
# Again I am well aware that tf does this, but I wanted to do it manually to practice.
########################################################################################################################

# finds the length of the longest title
def max_len(data):
    max_length = 0
    for temp_title in data:
        if len(temp_title) > max_length:
            max_length = len(temp_title)
    return max_length


# adds 0s to the end of titles until they're all the same length
def pad_sequences(train_data, valid_data):
    data = train_data + valid_data
    max_length = max_len(data)

    for temp_title in train_data:
        while len(temp_title) <= max_length:
            temp_title.append(0)

    for temp_title in valid_data:
        while len(temp_title) <= max_length:
            temp_title.append(0)

    for ind in range(len(train_data)):
        train_data[ind] = np.array(train_data[ind])
    for ind in range(len(valid_data)):
        valid_data[ind] = np.array(valid_data[ind])

    train_data = np.array(train_data)
    valid_data = np.array(valid_data)

    return train_data, valid_data, max_length


# Now we can pad the training and validation data sequences as well as convert the labels into numpy arrays.
training_data, validation_data, maxlen = pad_sequences(training_data, validation_data)

training_labels = np.array(training_labels)
validation_labels = np.array(validation_labels)

########################################################################################################################
# Now we can create the model.
# We define a few constants which we will use in the make_model function.
########################################################################################################################

buffer_size = 10000
batch_size = 32
embedding_dim = 64
vocab_size = len(token_dict) + 1
epochs = 5


########################################################################################################################
# Now we can make the model.
########################################################################################################################

# This function automates what activation to use.  Uses 'sigmoid' and binary_crossentropy if there are 2 labels
# and 'softmax' and 'sparse_categorical_crossentropy' otherwise.
def get_model_parameters(integer):
    if integer < 2:
        return 'eh?', 'eh?', 'eh?'
    if integer == 2:
        return 'sigmoid', 1, 'binary_crossentropy'
    if integer > 2:
        return 'softmax', integer, 'sparse_categorical_crossentropy'


custom_activation, custom_layer_number, custom_loss = get_model_parameters(len(list_of_labels))

# the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen + 1),
    tf.keras.layers.SpatialDropout1D(0.1),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(custom_layer_number, activation=custom_activation)
])

# prints summary
model.summary()

# compiles the model
model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])

# runs the fit, and stores the history for plotting
history = model.fit(
    x=training_data,
    y=training_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(validation_data, validation_labels),
)

########################################################################################################################
# This plots the accuracy and loss of the training and validation data over the epochs. (Optional)
########################################################################################################################

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


########################################################################################################################
# Now we can make the function which predicts the topic of a paper based on the title.  Note that out of token words
# are assigned token 0.
########################################################################################################################

# Converts the output of the NN on the test title into a string which can be presented to the user.
def topic_converter(cat_dict, classes):
    max_val = max(cat_dict.values())
    label_list = generate_list_labels(cat_dict, cross_topic)
    reverse_labels = {integer: tup for tup, integer in label_list.items()}

    if custom_activation == 'sigmoid':

        key_list = []
        for key in cat_dict.keys():
            key_list.append(key)
        p = classes[0][0]
        if p == 0.5:
            return 'ugh p = 0.5'
        if p > 0.5:
            return f'does NOT have topic in {key_list} with a probability {round(100 * p, 2)} %'
        if p < 0.5:
            return f'has topic {key_list} with probability {round(100 * (1 - p), 2)} %'

    else:

        key_list = []
        classes = classes[0]

        max_index = np.argmax(classes)
        max_prob = classes[max_index]

        labels = reverse_labels[max_index]
        labels = list(labels)

        for integer in labels:
            for key in cat_dict.keys():
                if cat_dict[key] == integer:
                    key_list.append(key)

        return f'has topic in {key_list} with probability {round(100 * max_prob, 2)} %'


# runs the model on the test title and outputs the verdict in plain english
def title_to_topic(model, cat_dict, string):
    # gets list of words from test title
    temp = improved_word_list(string, stop_words, df, 'outlier metric', cutoff)
    ttemp = []

    # pads the sequence
    for word in temp:
        if word in reverse_dict.keys():
            ttemp.append(reverse_dict[word])
        else:
            ttemp.append(0)
    while len(ttemp) <= maxlen:
        ttemp.append(0)

    # turns into array
    ttemp = np.array(ttemp)
    title = np.transpose(tf.expand_dims(ttemp, axis=-1))

    # runs the model and prints the output
    classes = model.predict(title, batch_size=10, verbose=0)
    print(classes)

    # converts the output into a plain english string about the most likely outcome
    topic_string = topic_converter(cat_dict, classes)  # predicts based on max probability

    # total string for output
    output_string = f'The paper \"{string}\" \n' + topic_string + '\n'

    return output_string


########################################################################################################################
# Now you can run this code with any title you want to see which topics the NN thinks the paper covers.
########################################################################################################################

# Example 1, a math paper
paper_title = "Tau function and moduli of meromorphic forms on algebraic curves"
guess = title_to_topic(model,
                       cat_dictionary,
                       paper_title)
print(guess)

# Example 2, a physics paper
paper_title = "Extended phase space thermodynamics of black hole with non-linear electrodynamic field"
guess = title_to_topic(model,
                       cat_dictionary,
                       paper_title)
print(guess)

# Example 3, a math paper
paper_title = "A formula for the categorical magnitude in terms of the Moore-Penrose pseudoinverse"
guess = title_to_topic(model,
                       cat_dictionary,
                       paper_title)
print(guess)

# Example 4, a mathematical physics paper
paper_title = "An entropic uncertainty principle for mixed states"
guess = title_to_topic(model,
                       cat_dictionary,
                       paper_title)
print(guess)

########################################################################################################################
################################################ Final Thoughts ########################################################
#
# With this model architecture and data, there is still immediate overfitting, independ on various hyperparameters
# I've tried.  I have a couple ideas about what's going on.
#
# 1) Use an LSTM layer to help with sequencing. I'm a little doubtful this will make a huge difference because
# my intuition tells me which words appear in the title matter more for topic categorization that what order they were
# in, but there could be cases where the order helps.  This brings me to the next point.
#
# 2) A very large data size is needed, or at least one with a very diverse set of buzzwords.  Consider, Example 3
# above.  On the one hand, I as a mathematician would recognize this as a category theory paper, and hence as a math
# paper, since I know that the Moore-Pensore pseudoinverse is a category theoretic phrase.
# However, the model had an output that it was a math/physics paper with a probability > .90, which is high.
# I suspect the model thought it was a math paper since it contained the words "formula", "categorical", and
# "pseudo-inverse", but also thought it was a physics paper since "Penrose" can refer to a famous physicist
# who probably gets name-dropped a few times in titles, and "magnitude" is also a common physics word.
# It would take a lot more data to tell the model that "Moore-Penrose pseudoinverse" is one thing that is firmly math.
# Perhaps more layers being added would help with situations like this.
#
# 3) This is just an inherently difficult problem with really noisy labels and vague titles from the metadata.
# Ultimately maybe many paper authors don't bother to cross label their papers even though a human reader
# may mistake them for papers from a different field.
# I would expect better results from the paper abstracts, but I wanted to see how far I could get with titles,
# since they were shorter, so analysis would be faster.
# All in all the first 2 epochs are usually pretty accurate and playing around with this data set has been
# fun, interesting, and good practice.
########################################################################################################################
