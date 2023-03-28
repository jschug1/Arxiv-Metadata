#########################################################################################
# This is a python file to export csv files which will be used in the
# plot_script R file.  The data frames we will export hold data on
# word usage in arxiv paper titles by year and by topic, as well as
# topic used per year.

import json
import sys
import pandas as pd
import numpy as np
import csv


#########################################################################################
# all the functions needed to process the full arxiv metadata
#########################################################################################

# creates two dictionaries out of a list of distinct elements.
# lst_dict associates a number to each element
# reverse_lst_dict associates the element to the number (or token)
def tokenize_list(lst):
    for i in range(len(lst)):
        lst[i] = [lst[i], i]

    lst_dict = {i: element for [element, i] in lst}
    reverse_lst_dict = {element: i for [element, i] in lst}

    return lst_dict, reverse_lst_dict


# remove_latex_eqns(str): takes a string of text and removes latex formatted equations from it
# it removes inline math formatting which are contained in $ ... $
# and maybe even formatted equations in [ ... ]
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
def word_list(text):
    text = remove_latex_eqns(text)
    chars_to_remove = ['(', ')', '.', ','] # list of punctuation to remove

    text = text.replace('\n', ' ') # removes next line char

    for char in chars_to_remove:
        text = text.replace(char, '') # removes the punctuation

    text = text.split(' ') # splits by spaces

    for i in range(len(text)):
        text[i] = text[i].lstrip() # removes left space
        text[i] = text[i].rstrip() # removes right spaces
        text[i] = text[i].lower() # makes lowercase

    # adds the nonempty words to a list
    words = []
    for word in text:
        if word != '':
            words.append(word)

    return words


# recovers the time that the paper first appeared on the arxiv
def paper_time(paper_dict):
    versions = paper_dict['versions']
    time = (versions[0])['created']

    time = time.split(' ')
    time[0] = time[0][:3]

    return time


# recovers the year from the output of paper_time
def published_year(paper):
    time = paper_time(paper)
    year = time[3]
    return year


# removes characters from lst from the string 'word'
def remove_chars(word, lst):
    new_word = word

    for string in lst:
        new_word = new_word.replace(string, '')

    return new_word


# sums up numbers in a list
def sum_over(num_list):
    temp_sum = 0
    for num in num_list:
        temp_sum += num
    return temp_sum


#########################################################################################
# Now we will generate a list of all subject IDs and years used in the dataset.
#########################################################################################


# this function generates all the tokens for the possible list of categories that can
# appear for a paper. There are 176 total I believe.  The tokens are easier to store
# than the big list of arxiv categories.
#
# It also keeps track of all the years that appear in the data set.
def generate_category_tokens():
    cat_list = [] # initializes list of categories
    year_list = [] # initializes list of years
    index = 0 # for progress tracking purposes in console

    for entry in open('arxiv-metadata.json', 'r'):

        value = float(index / 2207558) * 100  # for purposes of progress tracking
        print(f'generating category/year list, {value:.2f} %', end='\r')
        sys.stdout.flush()

        paper = json.loads(entry) # loads paper from json file

        categories = paper['categories']  # gets string of arxiv categories
        categories = categories.split(' ') # splits by space

        year = int(published_year(paper)) # integer year the paper first appeared on arxiv

        # appends distinct categories
        for cat in categories:
            if cat not in cat_list:
                cat_list.append(cat)

        # appends distinct years
        if year not in year_list:
            year_list.append(year)

        index += 1

    # sorts the years for simplicity
    year_list = sorted(year_list)

    # makes token dictionaries from tokenize_list function defined earlier
    cat_dict, reverse_cat_dict = tokenize_list(cat_list)
    yr_dict, reverse_yr_dict = tokenize_list(year_list)

    return cat_dict, reverse_cat_dict, yr_dict, reverse_yr_dict

# instantiates the data
category_dict, reverse_category_dict, year_dict, reverse_year_dict = generate_category_tokens()

# you can check the topic tokens if you're curious.
print(category_dict)
print(len(category_dict))

# Same for years, but this is not as necessary since the years are integers
print(year_dict)


#########################################################################################
# Now we can make a dataframe with the necessary information
#########################################################################################

# This generates the data frames we will export to csv files.
def generate_frames():
    words_by_year_dict = {} # initializes word by year dictionary
    topic_by_year_dict = {} # initializes topic by year dictionary
    words_by_topic_dict = {} # initializes words by topic dictionary
    total_by_year = {i: 0 for i, element in year_dict.items()} # some dictionaries for totals
    total_by_cat = {i: 0 for i, element in category_dict.items()}
    total_by_first_cat = {i: 0 for i, element in category_dict.items()}

    index = 0
    for entry in open('arxiv-metadata.json', 'r'):

        value = float(index / 2207558) * 100  # for purposes of progress tracking
        print(f'generating dicts, {value:.2f} %', end='\r')
        sys.stdout.flush()

        paper = json.loads(entry) # loads paper from json
        categories = paper['categories'] # list of categories from metadata
        categories = categories.split(' ') # split by space

        title = paper['title'] # title of paper as string
        title = word_list(title) # makes list of words from word_list function

        year = int(published_year(paper))   # integer year paper appeared on arxiv
        total_by_year[reverse_year_dict[year]] += 1 # adds to the total count for that year

        # adds each category appearance to the appropriate count in the appropriate dictionary
        for cat in categories:
            if cat in topic_by_year_dict.keys():
                topic_by_year_dict[cat][reverse_year_dict[year]] += 1 # +1 to topic year dict
            else:
                topic_by_year_dict[cat] = [0] * len(year_dict) # initializes key, value
                topic_by_year_dict[cat][reverse_year_dict[year]] += 1

        # turns cat list into list of tokens
        for i in range(len(categories)):
            categories[i] = reverse_category_dict[categories[i]]


        total_by_first_cat[categories[0]] += 1 # keeps track of primary arxiv category

        # adds to the total category count
        for cat in categories:
            total_by_cat[cat] += 1

        # now adds data for word appearance using word_list
        for word in title:
            if word in words_by_year_dict.keys():
                words_by_year_dict[word][reverse_year_dict[year]] += 1
            else:
                words_by_year_dict[word] = [0] * len(year_dict)

            if word in words_by_topic_dict.keys():
                for cat in categories:
                    words_by_topic_dict[word][cat] += 1
            else:
                words_by_topic_dict[word] = [0] * len(category_dict)
                for cat in categories:
                    words_by_topic_dict[word][cat] += 1

        index += 1

    # getting rid of words which appear < 100 times
    print("Restricting to common words")
    words_by_year_dict = {word: lst for word, lst in words_by_year_dict.items()
                          if sum_over(lst) > 100}
    words_by_topic_dict = {word: lst for word, lst in words_by_topic_dict.items()
                           if sum_over(lst) > 100}

    # converts to pandas data frames
    print("Converting to pandas.dataframes")
    df_words_by_year = pd.DataFrame(words_by_year_dict)
    df_topic_by_year = pd.DataFrame(topic_by_year_dict)
    df_words_by_topic = pd.DataFrame(words_by_topic_dict)

    df_year = pd.DataFrame(year_dict, index=['TOKEN_LABEL']).transpose()
    df_category = pd.DataFrame(category_dict, index=['TOKEN_LABEL']).transpose()

    df_year_total = pd.DataFrame(total_by_year, index=['YEAR_TOTAL']).transpose()
    df_category_total = pd.DataFrame(total_by_cat, index=['CAT_MENTION_TOTAL']).transpose()
    df_first_cat_total = pd.DataFrame(total_by_first_cat, index=['FIRST_CAT_TOTAL']).transpose()

    # adds on totals to data frame for less csv files
    df_words_by_year = pd.concat([df_year, df_year_total, df_words_by_year], axis='columns')
    df_topic_by_year = pd.concat([df_year, df_year_total, df_topic_by_year], axis='columns')
    df_words_by_topic = pd.concat([df_category, df_first_cat_total, df_category_total, df_words_by_topic],
                                  axis='columns')

    return df_words_by_year, df_topic_by_year, df_words_by_topic

# instantiates data frames
df_word_year, df_topic_year, df_word_topic = generate_frames()

# you can see the summary if curious
print(df_word_year)
print(df_topic_year)
print(df_word_topic)

#########################################################################################
# Exporting them to csv files
#########################################################################################

df_word_year.to_csv('word_year.csv')
df_topic_year.to_csv('topic_year.csv')
df_word_topic.to_csv('word_topic.csv')
