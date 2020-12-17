# CSE583: Natural Language Processing
### Zhan Shi, Neilly Herrera Tan, Angel Burr

## Procedural specification
### Background
Natural language processing affords new possibilities for sentiment analysis of text. This is especially beneficial in everyday scenarios when interacting with chatbots: for example, when asking Siri for weather updates or interfacing with a customer service bot.

To this end, we seek to investigate how to better understand natural language processing. Our main problem being addressed is how to generate fake text that effectively matches the style of an input corpus. This program will specifically generate novel poems based on trained datasets of poems by different authors.

### User profile
**Profile 1:** Someone who wants to generate any fake text and who wants to know a little bit more about natural language processing. This user will have a basic knowledge of Python, and is able to run the program using CLI. 

**Profile 2:** Another application is for users that have a bit of a sense of humor - these programs are really meant for fun and to explore how language develops. It is exciting to see what new patterns of language might form out of using NLP toolkits and deep learning programs. Depending on how refined the model is, often times these programs don't make total sense, but are entertaining and interesting nonetheless.

_Scenario:_ An aspiring poet herself, Dara is Emily Dickinson's number one fan. She has read through all of Emily Dickinson's poems. However, Dara wants to read more, and wants to know how to incorporate a similar Dickinsonian style to her own poetry. To figure out the essence of Emily Dickinson's poems, and to read more work that emulates Emily Dickinson's work, Dara runs this program with sample texts from her favorite Dickonson poems. 

### Data sources
We will use input data from a collection of poems via Project Gutenberg, on their [poetry bookshelf](http://www.gutenberg.org/ebooks/bookshelf/60).
* [Poems by TS Eliot](http://www.gutenberg.org/cache/epub/1567/pg1567.txt)
* [Poems by John Keats](http://www.gutenberg.org/cache/epub/2490/pg2490.txt)
* [Poems by Emily Dickinson](http://www.gutenberg.org/cache/epub/2678/pg2678.txt)

### Use case
The main use case for this program is for generating fake text. Our program takes an input of some text, and outputs a wall of fake text with similar sentiment to the input text. 
* **User input:** A few paragraphs of text such as poems from various authors. 
* **Output:** Computer generated fake text that mimics the style of the input text.

This text generator program is meant for fun. It is designed for educational and experimental purposes. With a more advanced program, a person may be able to generate greeting cards or fortune cookies based on larger or smaller text inputs. Additionally, this program can be used for more nonsensical cases in the future, such as a NLP twitter bot that characterizes the voices of old poets, or for fun purposes in an art installation. These examples are implications for future work. For class purposes, our program may not be advanced enough to generate meaningful text in this manner.