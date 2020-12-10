# NLP-Project

## Background
Natural language processing affords new possibilities for sentiment analysis of text. This is especially beneficial in everyday scenarios when interacting with any chatbot: for example, when asking Siri for weather updates or interfacing with a customer service bot.

To this end, we seek to investigate how to better understand natural language processing. Our main problem being addressed is how to generate fake text that effectively matches the style of an input corpus. This program will specifically generate novel poems based on trained datasets of poems by different authors.

## Team Members
* Angel Burr
* Neilly Herrera Tan
* Zhan Shi

## Installation
Here are instructions to locally install this program. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) if you have not already done so.
1. Clone the repository: `git clone https://github.com/neillytan/NLP-Project.git`
2. Create a conda environment: `conda env create -q -n nlp-project --file environment.yml`
3. Activate the environment: `conda activate nlp-project`
4. Run the jupyter notebook demo: `jupyter notebook demo.ipynb` --> *See note below*
5. Once you have opened the notebook, replace the `input` variable with your own text file. Then, input your desired output word length in the variable `seq_length` for the gready decoder. 

**Note:** We are working to export the Jupyter notebook into separate modules. These instructions will change in the upcoming week. 

## Directory Summary
The directory has folders for docs and NLP (which includes our source code).

## Directory Structure
```
NLP-Project/
├── NLP
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-37.pyc
│   └── src
│       ├── pre_processing
│       │   ├── __init__.py
│       │   ├── pre.py
│       │   ├── pre_processing.py
│       │   └── text.txt
│       └── training
│           ├── __init__.py
│           ├── __pycache__
│           │   ├── __init__.cpython-37.pyc
│           │   ├── training.cpython-37.pyc
│           │   └── utils.cpython-37.pyc
│           ├── model.py
│           ├── training.py
│           └── utils.py
├── README.md
├── demo.ipynb
├── docs
│   ├── CSE583_tech_review.pdf
│   ├── component-flowchart.jpeg
│   ├── component-specs.ipynb
│   └── procedural-specs.ipynb
├── requirements.txt
└── src


```

## Project Data 
For this project, we are using text from Project Gutenburg, specifically a collection of poems by Emily Dickinson and John Keats.

## Project History 
This project was made in Autumn 2020 as part of the CSE 583 course at the University of Washington.

## Future Work
If time allows, we aim to add a Twitter bot to this program. 