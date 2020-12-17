## CSE583: Natural Language Processing
#### Zhan Shi, Neilly Herrera Tan, Angel Burr

Component flowchart:
<img src= "component-flowchart.jpeg">
Here, the pipeline is broken down into three main parts: Preprocessing, Training, and Decoding. Generally, the program takes an input corpus ``input`` (any .txt file), and generates a word list ``output``. Here, the input is passed into a vectorizer that maps the words onto vectors during the Preprocessing stage. Then, the input will train the language model (Training), and decode this model (Decoding) to generate a fake text output.

#### Preprocessing
Using NLTK, preprocess the input text with 4 main functions in mind:
1. Tokenize the input.
2. Indexing
3. Generate ``EndOfSentence`` tokens
4. Break token list into pieces based on ``EOS`` tokens for batch training

#### Training
Train the language model with the input given, containing two parts:
1. Initializing the embedding matrix with pre-trained vectors (using gensim) when available, otherwise start at random.
2. Train the LSTM model

#### Decoding
Decode the language model with pytorch to generate fake text. This uses both the greedy method and top k random sampling method.