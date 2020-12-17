# NLP-Project
[![Build Status](https://travis-ci.com/travis-ci/travis-web.svg?branch=master)](https://travis-ci.com/travis-ci/travis-web)
[![Coverage Status](https://coveralls.io/repos/github/neillytan/NLP-Project/badge.svg?branch=main)](https://coveralls.io/github/neillytan/NLP-Project?branch=main)

## Background
Natural language processing affords new possibilities for sentiment analysis of text. This is especially beneficial in everyday scenarios when interacting with any chatbot: for example, when asking Siri for weather updates or interfacing with a customer service bot.

To this end, we seek to investigate how to better understand natural language processing. Our main problem being addressed is how to generate fake text that effectively matches the style of an input corpus. This program will specifically generate novel poems based on trained datasets of poems by different authors.

## Team Members
* Angel Burr
* Neilly Herrera Tan
* Zhan Shi

## Running the program
Here are instructions to locally install this program. 
1. Clone the repository: `git clone https://github.com/neillytan/NLP-Project.git`
2. Install the requirements: ``pip install -r requirements.txt``
3. Navigate to the examples folder `cd examples`. Here, you will see example demos of how to run this program and along with our explanation of the NLP training model.
4. Run the jupyter notebook demo `jupyter notebook Demo_.ipynb` or follow the html `open Demo.html`
5. Once you have opened the notebook or html page, replace the input variable (`sample_txt.txt`) with your own text file. Then, you can specify some options for you output text in code under the header, `Decoder Options`. Here, you can specify different output lengths for your NLP generated poem, for example.

## Directory Summary
The directory has folders for docs and NLP (which includes our source code).

## Directory Structure
```
NLP-Project
.
├── LICENSE
├── NLP
│   ├── __init__.py
│   ├── data
│   │   ├── sample-txt-2.txt
│   │   ├── sample-txt-3.txt
│   │   └── sample_txt.txt
│   └── src
│       ├── decoder
│       │   ├── __init__.py
│       │   └── decoder.py
│       ├── generator
│       │   ├── __init__.py
│       │   └── generator.py
│       ├── pre_processing
│       │   ├── __init__.py
│       │   └── pre_process.py
│       └── training
│           ├── __init__.py
│           ├── model.py
│           ├── training.py
│           └── utils.py
├── README.md
├── __init__.py
├── docs
│   ├── CSE583_tech_review.pdf
│   ├── component-specs.ipynb
│   ├── component-specs.md
│   ├── img
│   │   └── component-flowchart.jpeg
│   └── procedural-specs.md
├── examples
│   ├── Demo.html
│   ├── Demo_.ipynb
│   └── demo_old.ipynb
├── requirements.txt
├── setup.py
└── tests
    ├── __init__.py
    ├── test_decoder_end_to_end.py
    ├── test_generator_end_to_end.py
    ├── test_pre_process.py
    ├── test_training_end_to_end.py
    ├── test_training_utils.py
    └── text.txt
```

## Project Data 
For this project, we are using text from Project Gutenburg, specifically collections of poems by T.S. Eliot, Emily Dickinson, and John Keats.
* [Poems by TS Eliot](http://www.gutenberg.org/cache/epub/1567/pg1567.txt) ``NLP/data/sample-txt.txt``
* [Poems by John Keats](http://www.gutenberg.org/cache/epub/2490/pg2490.txt) ``NLP/data/sample-txt-2.txt``
* [Poems by Emily Dickinson](http://www.gutenberg.org/cache/epub/2678/pg2678.txt) ``NLP/data/sample-txt-3.txt``

## Project History 
This project was made in Autumn 2020 as part of the CSE 583 course at the University of Washington.

## Limitations
This repository will not be maintained after 2020.  