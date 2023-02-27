########################################################################################
# make_arxiv_data - inputs the file name and outputs a list of paper info from the arxiv-metadata.json file from Kaggle
#                   could be adapted to a subset of papers, such as the ones which contain 'math' in the category
#                   if such a json file were made
#
# input: str = file name as a string
#
# output: list of dictionaries, one for each paper in the arxiv data
########################################################################################

def make_arxiv_data(str):
    arxiv_data = []
    index = 0
    for paper in open(str, 'r'):
        arxiv_data.append(json.loads(paper))
        if index % 100000 == 0:
            print(index)
        index += 1

    return arxiv_data


########################################################################################
# make_published_dict - input a list of dictionaries, typically the output of make_arxiv_data, and makes a dictionary
#                       of whether it was in a journal or not
#
# input: data = list of dictionaries with paper info
#
# output: published_dict = dictionary with id's and a value of 0 or 1 if it was published or not
########################################################################################

def make_published_dict(data):
    published_dict = {}
    for paper in data:
        published_dict['id'] = paper['id']
        if paper['journal-ref'] == None:
            published_dict['in-journal'] = 0
        else:
            published_dict['in-journal'] = 1

    return published_dict


########################################################################################
# make_time_dict: makes a dictionary of papers recording the time of the first version uploaded to the arxiv
#                   references the next function, paper_time
# input: data = the list of dictionaries for all the desired papers you wish to parse
#
# output: time_dict = dictionary with the 'time' value being the time of the upload of the first version of the paper
########################################################################################

def make_time_dict(data):
    time_dict = {}
    for paper in arxiv_data:
        time_dict['id'] = paper['id']
        time_dict['time'] = paper_time(paper)

    return time_dict


########################################################################################
# paper_time: picks out the time of the first version upload from the paper dictionary.
#             Dependent on formatting of the arxiv-metadata.json dataset
#
# input: paper_dict = dictionary of paper data, including a 'versions' value.
#
# output: time = the time of the first upload of the paper to the arxiv, as a list of time units
########################################################################################

def paper_time(paper_dict):
    versions = paper_dict['versions']
    time = (versions[0])['created']

    time = time.split(' ')
    time[0] = time[0][:3]

    return time


########################################################################################
# all_caps: returns True if word is in all caps, and False otherwise.  Not really used, but I wrote it as an exercise
#
# input: str = string of only letters to be checked for all caps
#
# output: Boolean value, True if all caps, False if contains a lowercase letter
########################################################################################

def all_caps(str):
    for letter in str:
        if letter.islower():
            return False
        else:
            continue

    return True


########################################################################################
# only_letters: checks if a string has only letters in it, again not really used, just a toy function
#               definitely not my finest function
#
# input: str = string to be checked for only letters
#
# output: Boolean, False if contains a number or symbol, True if only letters
########################################################################################

def only_letters(str):
    for letter in str:
        if letter in '0123456789!#$%^&*(){[}]:;",<.>/?|\'\n':
            return False
        else:
            continue
    return True


########################################################################################
# restrict_chars: takes a string 'word', and a string of characters 'str' you wish to only include in your word.
#                   Ouputs 'word' with only the characters from 'str' in it
# input: word = a string that you wish to restrict the characters of
# input: str = a string of characters you want into include in your word
#
# output: str = a string of characters you wish to include in your word, maybe only lowercases, so 'abcde...' etc
########################################################################################

def restrict_chars(word, str):
    new_word = ''
    for letter in word:
        if letter in str:
            new_word += letter

    return new_word


########################################################################################
# remove_chars: takes a list of strings and removes them from your word.  Just a multi input version of .replace( -, '')
#
# input: word = a string that you want to remove the characters from
# input: list = a list of strings you wish to get of from word
#
# ouput: new_word = word but with the strings from list removed
########################################################################################

def remove_chars(word, list):
    new_word = word

    for str in list:
        new_word = new_word.replace(str, '')

    return new_word


########################################################################################
# remove_parantheses: takes a string and removes the parentheticals '( ... )' from it, assuming correct grammar
#                       perhaps a slight misnomer, it does not remove only the '(' and ')', it will also remove
#                       everything in between those
#
# input: str = a string of text you wish to remove the parentheticals from
#
# output: newstr = a string of text with the parentheticals removed
########################################################################################

def remove_parenetheses(str):
    parenth = 0
    newstr = ''
    for letter in str:
        if letter == '(':
            parenth += 1
        if parenth == 0:
            newstr += letter
        if letter == ')':
            parenth -= 1

    return newstr


########################################################################################
# remove_latex_eqns(str): takes a string of text and removes latex formatted equations from it
#                           it removes inline math formatting which are contained in $ ... $
#                           and maybe even formatted equations in [ ... ]
# NOTE: a bit of a naive algorithm, and made for abstracts which typically only include inline math formatted as $ ... $
#
# input: str = string of text you wish to remove latex formatted equations from
#
# output: newstr = a string of text without the latex formatted equations
########################################################################################

def remove_latex_eqns(str):
    dolla = -1
    eqn = 0
    newstr = ''
    for letter in str:
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


########################################################################################
# decide_delimiter: takes a plain english list of names and picks out the way they are formatted
#                   used to pick out the family names from the list of authors in the arxiv-metadata.json file
#
# input: str = string of plain english list of names, usually 'author' value from the arxiv-metadata.json file
#
# output: one of 'one word', 'new rule', 'and', 'old rule', and ',' depending on how the list is formatted
#               single author -> one word
#               name 1, name 2, and name 3 -> new rule
#               name 1 and name 2 -> and
#               name 1, name 2 and name 3 -> old rule
#               name 1, name 2, name 3 -> ,
########################################################################################

def decide_delimiter(str):
    if ',' not in str and ' and ' not in str:
        return 'one word'

    if ', and' in str:
        return 'new rule'

    if ',' not in str and ' and ' in str:
        return 'and'

    if ',' in str and ' and ' in str:
        return 'old rule'

    if ' and ' not in str and ',' in str:
        return ','


########################################################################################
# name_process: pre-processing for a name, removes latex formatted accents, such as umlauts, aigus, etc
#
# input: str = name to pre-process
#
# output: str = name without latex formatted accents
########################################################################################

def name_process(str):
    str = str.replace('\\c{c}', 'c')
    str = str.replace('{\\l}', 'l')
    str = str.replace('{\\O}', 'O')

    str = str.lstrip()
    str = str.rstrip()

    extras = ['\\c', '\\"', "\\'", '\\`']
    str = remove_chars(str, extras)

    return str

########################################################################################
# custom_author_split: splits a plain english list of names based on the delimiter,
#                       which is ouputted from decide_delimiter
#
# input: authors = plain english list of names to pick out the names from
#
# output: split = a python list of authors from the plain english list 'authors'
########################################################################################

def custom_author_split(authors):
    delimiter = decide_delimiter(authors)

    if delimiter == 'one word':
        return [authors]

    if delimiter == ',':
        return authors.split(', ')

    if delimiter == 'and':
        return authors.split(' and ')

    if delimiter == 'old rule':
        first_split = authors.split(' and ')
        second_split = first_split[0].split(', ')
        second_split.append(first_split[-1])
        return second_split

    if delimiter == 'new rule':
        split = authors.split(', ')
        split[-1] = split[-1][5:]
        return split

########################################################################################
# last_name: given a name of a human, picks out the family name based on western name order,
#              which the arxiv-metadata.json uses, e.g. first name then last name
#               takes into account middle names or initials listed
#               with in-name prepositions like 'de' and 'von' accounted for
#
# input: str = string consisting of a human name
#
# output: last_name = string consisting of just the family name
########################################################################################

def last_name(str):
    euro = ['de', 'De', 'Van', 'van', 'da', 'Da', 'del', 'Del', 'von', 'Von']
    last_name = ''

    number_of_names = len(str.split(' '))

    if number_of_names == 1:
        temp_name = str.split('.')
        last_name = temp_name[-1]

    if number_of_names == 2:
        temp_name = str.split(' ')
        last_name = temp_name[-1]

    if number_of_names > 2:
        temp_name = str.split(' ')
        if temp_name[-2] in euro:
            last_name = ' '.join(temp_name[-2:])
        else:
            last_name = temp_name[-1]

    return last_name

########################################################################################
# find_author_names: given a string of text, typically academic writing, and picks out the names of people mentioned
#
#                       a couple of exceptions are accounted for including
#                       - possessives are removed
#                       - adjectives with 'ian', like 'Riemannian', are counted as the name, e.g. 'Riemann'
#                       - 'abelian' or 'noetherian' aren't capitalized, so those are treated separately
#                           as 'Abel' and 'Noether'
#                       - sometimes names occur with a dash, i.e. 'Navier-Stokes', the dash is removed
#                           and Navier and Stokes are accounted for separately
#                       - the dash algorithm is a bit complicated since adjectives like 'strong-XXX' may appear
#                           where XXX is a name we want to account for
#                       - currently the first word is ignored, so hopefully no one opened their text with a proper name
#                       - currently does not account for initialisms like WDVV which may or may not include a name
#
# input: abstract = a string of text, usually from academic writing
#
# output: names = a list of names of people referenced in the text
########################################################################################

def find_author_names(abstract):
    restrict = ['Conjecture', 'Lemma', 'Theorem', 'Corollary', 'PDE', 'TV', 'Functional', 'Processor', 'System', 'We',
                'The', 'This', 'In', 'If', 'Random']
    extras = ['\\c', '\\"', "\\'", '\\`']

    # print(abstract)

    # abstract = remove_latex_eqns(abstract)
    # print(abstract)

    abstract = abstract.replace('\n', ' ')
    abstract = abstract.split(' ')
    # print(f'list abstract: {abstract}')
    for i in range(len(abstract)):
        abstract[i] = abstract[i].lstrip()
        abstract[i] = abstract[i].rstrip()

    for word in reversed(abstract):
        if word == '':
            abstract.remove(word)

    names = []
    for i in range(len(abstract)):
        if abstract[i] == 'abelian':
            names.append('Abel')
        if abstract[i] == 'noetherian':
            names.append('Noether')
        if '-' in abstract[i] and len(abstract[i]) > 1:
            adj = abstract[i].split('-')
            for word in reversed(adj):
                if word == '':
                    adj.remove(word)
            for word in adj:
                if word[0].isupper() and len(word) > 1:
                    names.append(word)
                    continue
            continue
        if abstract[i][0].isupper():
            if abstract[i - 1][-1] not in ':.':
                names.append(abstract[i])
                continue

    # print(f'caps = {names}')

    for i in range(len(names)):
        if names[i][-2:] == "'s":
            names[i] = names[i][:-2]
            continue
        if names[i][-1] == "'":
            names[i] = names[i][:-1]
            continue
        if names[i][-1] == '.':
            names[i] = names[i][:-1]
            continue

    for name in reversed(names):
        if len(name) < 2:
            names.remove(name)
        for char in ":.,$_-^\'')":
            if char in name:
                names.remove(name)
                break
        if name == 'Rham':
            names.remove(name)
            names.append('De Rham')
        if name in restrict:
            names.remove(name)

    # print(f'names = {names}')
    for i in range(len(names)):
        # print(f'i = {i}')
        # print(f'name = {names[i]}')

        if names[i] == 'Gaussian':
            names[i] = 'Gauss'
        if names[i] == 'Riemannian':
            names[i] = 'Riemann'
        if names[i] == 'Abelian':
            names[i] = 'Abel'
        if names[i] == 'Noetherian':
            names[i] = 'Noether'
        if names != []:
            names[i] = remove_chars(names[i], extras)

    return names

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
# random_list_of_papers: generates a random list of paper dictionaries from the arxiv-metadata.json dataset
#
# input: number = an integer number of papers you want to sample from the dataset
#
# output: sample_papers = a list of length 'number' of random papers from arxiv-metadata.json
########################################################################################

def random_list_of_papers(number):
    import random
    randomlist = random.sample(range(2207558), number)

    sample_papers = []
    index = 0
    for entry in open('arxiv-metadata.json', 'r'):
        if index in randomlist:
            paper = json.loads(entry)
            sample_papers.append(paper)
        index += 1

    return sample_papers

########################################################################################
# published_year: picks out the year from the paper_time function, so the year the paper was first uploaded to arxiv
#
# input: paper = a dictionary of paper metadata from arxiv-metadata.json
#
# output: year = the year the first version appeared on the arxiv
########################################################################################

def published_year(paper):
    time = paper_time(paper)
    year = time[3]
    return year

########################################################################################
# convert_ltd: converts a list to a dictionary with keys distinct elements from the list and values the number of time
#                that key appears in the list
#
# input: list = the list to convert into a count dictionary
#
# output: dict = the dictionary of counts of distinct entries from list
########################################################################################

def convert_ltd(list):
    dict = {}
    for element in list:
        if element in dict.keys():
            dict[element] += 1
        else:
            dict[element] = 1

    return dict

########################################################################################
# update_dict: combines two dictionaries with integer values, adds the integer values for equal keys
#
# input: dict1 = a dictionary to combine
# input: dict2 = a dictionary to combine
#
# output: dict3 = combined dictionary
########################################################################################

def update_dict(dict1, dict2):
    dict3 = {words: dict1.get(words, 0) + dict2.get(words, 0) for words in set(dict1).union(dict2)}

    return dict3

########################################################################################
# make_total_mention_graph: makes a graph of total mentions of a name from the abstract of arxiv-metadata.json
#                           references the mentions-by-year.json file generated in this project
#
# input: researcher = a string of a family name
#
# output: years = a numpy array of the years with arxiv metadata with first five years removed for small sample size
# output: count = a numpy array of the counts of the mentions of the input with data from mentions-by-year.json
# output: will also show a pyplot plot of (years, count) = (x,y)
########################################################################################

def make_total_mention_graph(researcher):
    years = []
    count = []
    for entry in open('mentions-by-year.json', 'r'):
        data = json.loads(entry)

    i = 0
    for year in data.keys():
        mentions = data[year]
        years.append(year)
        count.append(0)
        if researcher in mentions.keys():
            count[i] = mentions[researcher]
        i += 1
    data = []

    combo = []
    for i in range(len(years)):
        combo.append([years[i], count[i]])

    combo = sorted(combo, key=lambda x: x[0])

    years = []
    count = []
    for i in range(len(combo)):
        years.append(int(combo[i][0]))
        count.append(combo[i][1])

    years = np.array(years)
    count = np.array(count)

    years = years[5:]
    count = count[5:]

    plt.title('Total Mentions in Abstract vs Year')
    plt.xlabel('Year')
    plt.ylabel('Mentions')
    plt.plot(years, count)
    plt.show()

    return years, count

########################################################################################
# year_numb_array: makes an np array of the number of papers appearing in the year according to chronological order
#
# input: filename = a string of the filename, typically 'arxiv-metadata.json' but maybe one made a similarly formatted json file
#
# output: counts = an np array of the number of papers from that year, with year counted in chronological order from 1986
########################################################################################

def year_numb_array(filename):
    number_year_dict = {}
    for entry in open(filename, 'r'):
        paper = json.loads(entry)
        year = published_year(paper)
        if year in number_year_dict.keys():
            number_year_dict[year] += 1
        else:
            number_year_dict[year] = 1

    sorted_nyd = sorted(number_year_dict.items(), key=lambda x: x[0])
    counts = []
    for entry in sorted_nyd:
        counts.append(entry[1])

    # print(sorted_nyd)

    return counts


########################################################################################
# make_relative_word_graph: makes a plot of the number of mentions of the word in the abstract,
#                                   relative to the number of papers published that year
#                                   depends on the output of the year_numb_array function
#                                   need NOT be a name
#
# input: word = a string
#
# output: years = an np array of years that papers have listed from published_year in chronological order,
#                   first 5 years removed for small sample size
# output: count = an np array of counts of the number of mentions of 'word' in the abstracts from papers of
#                   that year
# output: also plots a pyplot of (years, count) = (x,y)
########################################################################################

def make_relative_word_graph(word):
    years = []
    count = []
    for entry in open('words-by-year.json', 'r'):
        data = json.loads(entry)

    word = word.lower()

    i = 0
    for year in data.keys():
        mentions = data[year]
        years.append(year)
        count.append(0)
        if word in mentions.keys():
            count[i] = mentions[word]
        i += 1
    data = []

    combo = []
    for i in range(len(years)):
        combo.append([years[i], count[i]])

    combo = sorted(combo, key=lambda x: x[0])

    years = []
    count = []
    for i in range(len(combo)):
        years.append(int(combo[i][0]))
        count.append(combo[i][1])

    years = np.array(years)
    count = np.array(count)
    count = 100 * count / total_papers_year

    years = years[5:]
    count = count[5:]

    plt.title('Relative Mentions in Abstract vs Year')
    plt.xlabel('Year')
    plt.ylabel('Mentions %')
    plt.plot(years, count)
    plt.show()

    return years, count
