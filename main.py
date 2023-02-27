import numpy as np
import matplotlib.pyplot as plt

import json
from functions import *

total_papers_year = year_numb_array('arxiv-metadata.json') # makes the array containing the number of papers for that year
make_relative_word_graph('higgs')
# an example graph, you can see an uptick in mention rate around when the Higgs boson was experimentally verified in 2012
make_relative_word_graph('Navier-Stokes')
# another interesting example, people are generally more interested in Navier-Stokes over the years
make_relative_word_graph('relativity')
make_relative_word_graph('GR')
# this is interesting because 'relativity' is mentioned less, but the initialism 'GR' is generally increasing in use over time