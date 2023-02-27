This project deals with the arxiv-metadata.json dataset available here: https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download.
You'll need to download it first to run this code.  In this first version of the project, I've implemented a function which visualizes
word mention percentage within the abstracts. Then one can potentially see how interest in a topic or a persons work changes over time just through the metadata.
Some interesting examples are mentions of 'Higgs' or 'Navier-Stokes'.

In order to run the functions, you first need to generate the words-by-year.json file.  The code to do this is dict_gen.py.
Then you can run the main.py.  The functions are defined in functions.py.  That document contains some extra functions I was working on,
and my use in the future.  The main function used is make_relative_word_graph.