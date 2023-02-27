########################################################################################
# This document contains code which generates the words-by-year.json file from arxiv-metadata.json.
# The words-by-year.json file is a nested dictionary with first level keys given by the years 1992 - 2023,
# second level keys are of words which appeared in abstracts that year, and the values are the numer of times
# that word appeared in abstracts that year
########################################################################################

index = 0
words_dict = {} # initializes the dictionary of words.
for entry in open('arxiv-metadata.json', 'r'): # opens arxiv-metadata
    paper = json.loads(entry) # loads the dictionary of info for paper

    if index%10000 == 0:
        print(f'index = {index}')

    abstract = paper['abstract'] # opens abstract data
    abstract_words = word_list(abstract) # parses it into a list of lower case words
    abstract_words = list(set(abstract_words)) # only takes distinct words since we are measuring mention rate

    year = int(published_year(paper)) # gets year of first version

    if year in words_dict.keys(): # adds the words from the abstract to a larger dictionary according to year
        words_dict[year] += abstract_words
    else:
        words_dict[year] = abstract_words

    index += 1

for year in words_dict.keys(): # turns the list of words from abstracts that year into a dictionary of counts
    print(f'year = {year}')
    words_dict[year] = convert_ltd(words_dict[year])


filename = 'words-by-year.json' # stores the dictionary as a json file
json_dump = json.dumps(words_dict)
with open(filename, "w") as outfile:
    outfile.write(json_dump)
words_dict = {} # resets the dictionary so the console doesn't store the large dataset as a variable long-term

