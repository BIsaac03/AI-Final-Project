######################################################
# Programmer:       Benjamin Isaac (bmi22)
# Class:            Artificial Intelligence (CSE 4633)
# Professor:        Eric Hansen
# University:       Mississippi State Univeristy
# Last Modified:    12/01/2023
#
#
# External Sources Consulted:
#
# Setence datasets from: https://tatoeba.org/en/downloads
#
# Extensive use of TensorFlow webpages, particularly:
#       https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
#       https://www.tensorflow.org/tutorials/keras/text_classification, 
#       https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization
#
# https://www.kaggle.com/code/martinkk5575/language-detection/notebook
#
#
#
######################################################

import tensorflow as tf
import pandas as pd
import numpy as np
from keras import layers

# maps model outputs to the languages they represent
languageIndices = {0: 'English', 1: 'Spanish', 2: 'French', 3: 'Portuguese', 4: 'Indonesian', 5: 'German', 6: 'Turkish', 7: 'Vietnamese', 8: 'Italian', 9: 'Hungarian'}

print("Welcome to the language detector")
print("The following languages are supported:\n")
print("1. English")
print("2. Spanish")
print("3. French")
print("4. Portuguese")
print("5. Indonesian")
print("6. German")
print("7. Turkish")
print("8. Vietnamese")
print("9. Italian")
print("10. Hungarian")

# loads trained model
model = tf.keras.models.load_model('language_probability_model.keras')

sentenceTrainingData = pd.read_csv('sentenceTrainingData.tsv', sep='\t', names=["Sentence"])
vectorize_layer = layers.TextVectorization()
vectorize_layer.adapt(sentenceTrainingData)

history = False                         # checks if any text has been entered into the program
keepGoing = True                        # checks if the program should continue running

# loops until user chooses to leave
while keepGoing:
    text = input("\nEnter a sentence or phrase to analyze. To view more details about the previous entry, enter '+'. To exit the program, enter 'EXIT'.  ")

    # quits program
    if text == 'EXIT':
        keepGoing = False

    # expands on details of previous entry
    elif text == '+':
        if history == False:
            print("\nYou must first enter text from one of the supported languages.")

        else:
            print("\nThe confidence levels for other languages are:\n")

            # marks best guess as already printed
            prediction[0][guess] = -1

            # loops until all confidence levels have been printed
            while prediction[0][np.argmax(prediction[0])] != -1:

                # prints next best guess
                guess = np.argmax(prediction[0])
                print("{0}- {1:.2f}% confidence".format(languageIndices[guess], prediction[0][guess] * 100))

                # marks next best guess as already printed
                prediction[0][guess] = -1
        
    # analyzes new entry
    else:
        # allows for future '+' info
        history = True

        # vectorizes text
        text = tf.expand_dims(text, -1)
        data = vectorize_layer(text)

        # uses probability model to predict the text's language of origin
        prediction = model.predict(data, verbose = 0)

        # prints best guess
        guess = np.argmax(prediction[0])
        print("The text is {0} ({1:.2f}% confidence)".format(languageIndices[guess], prediction[0][guess] * 100))