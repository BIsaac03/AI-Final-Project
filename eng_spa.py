import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.layers import TextVectorization


from tensorflow import keras
from keras import layers
from keras import losses
from keras.models import Model
from keras.layers import Dense

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

#print("Tenserflow version: ", tf.__version__)
#print("")

#print("Pandas version: ", pd.__version__)
#print("")

# raw dataset of numbered sentences and corresponding language
engData = pd.read_csv('eng_sentences.tsv', sep='\t', names=["Number", "Language", "Sentence"])

# removes number, leaving just sentence and corresponding language
engData.pop("Number")
engData.pop("Language")

language = []

for i in range(len(engData)):
    language.append(0)

engData['Language'] = language

print("English data: \n", engData)

engData = engData.truncate(after = 9999)

# raw dataset of numbered sentences and corresponding language
spaData = pd.read_csv('spa_sentences.tsv', sep='\t', names=["Number", "Language", "Sentence"])

# removes number, leaving just sentence and corresponding language
spaData.pop("Number")
spaData.pop("Language")

language = []

for i in range(len(spaData)):
    language.append(1)

spaData['Language'] = language

print("Spanish data: \n", spaData)

spaData = spaData.truncate(after = 9999)

frames = [engData, spaData]
combinedData = pd.concat(frames)
print("Combined data: \n", combinedData)

print("Language data: \n", combinedData['Language'].value_counts())

# splits dataset into training (80%) and testing (20%) data
trainingData, testingData = train_test_split(combinedData, test_size=0.2)

print("Training data: \n", trainingData)
print("Testing data: \n", testingData)


sentenceTrainingData = trainingData.copy()
sentenceTrainingData.pop("Language")

languageTrainingData = trainingData.copy()
languageTrainingData.pop("Sentence")

sentenceTestingData = testingData.copy()
sentenceTestingData.pop("Language")

languageTestingData = testingData.copy()
languageTestingData.pop("Sentence")


#sentenceTrainingData = trainingData.copy()
#languageTrainingData = sentenceTrainingData.pop("Language")

#sentenceTestingData = testingData.copy()
#languageTestingData = sentenceTestingData.pop("Language")


trainingData = tf.data.Dataset.from_tensor_slices((sentenceTrainingData, languageTrainingData))
testingData = tf.data.Dataset.from_tensor_slices((sentenceTestingData, languageTestingData))

print("Training data elements: \n", trainingData.cardinality().numpy())
print("Testing data elements: \n", testingData.cardinality().numpy())

vectorize_layer = layers.TextVectorization()
sentenceTrainingData = trainingData.map(lambda x, y: x)
vectorize_layer.adapt(sentenceTrainingData)

text_batch, label_batch = next(iter(trainingData))
first_review, first_label = text_batch, label_batch
print("Review", first_review)
print("Label", first_label)
print("Vectorized review", vectorize_text(first_review, first_label))


train_ds = trainingData.map(vectorize_text)
test_ds = testingData.map(vectorize_text)


#text_batch, label_batch = next(iter(trainingData))
#sentence, language = text_batch[0], label_batch[0]
#print("Sentence", sentence)
#print("Label", language)
#print("Vectorized review", vectorize_text(sentence, language))


AUTOTUNE = tf.data.AUTOTUNE

trainingData = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
testingData = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#model = Sequential()
#model.add(layers.LSTM(22, input_shape=(7953,1)))
#model.add(Dense(22, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())
#model.fit(trainingData, epochs=2, batch_size=500)

model = Sequential([
 layers.Embedding(20000, 16),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)
  ])

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

model.summary()

# works to here

history = model.fit(trainingData, epochs=3, batch_size=50)

loss, accuracy = model.evaluate(testingData)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
loss = history_dict['loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


#export_model = tf.keras.Sequential([
#  vectorize_layer,
#  model,
#  layers.Activation('sigmoid')
#])

#export_model.compile(
#    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
#)

# Test it with `raw_test_ds`, which yields raw strings
#loss, accuracy = export_model.evaluate(testingData)
#print(accuracy)

#export_model.predict("ek het")