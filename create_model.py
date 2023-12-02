import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split

# takes a given .tsv file and its language's corresponding index and returns a pandas dataframe with 1000 sentence/languageIndex pairings
def preprocessData(file, languageIndex):

    # reads file into pandas dataframe
    dataframe = pd.read_csv(file, sep='\t', names=["Number", "Language", "Sentence"])

    # shortens to 1000 sentences
    dataframe = dataframe.truncate(after = 999)

    # removes unnecessary column
    dataframe.pop("Number")
    
    # converts language label to corresponding integer
    dataframe.pop("Language")
    language = []

    for i in range(1000):
        language.append(languageIndex)

    dataframe['Language Index'] = language

    # returns resulting dataframe
    return dataframe

# applies text vectorization layer
def vectorize_text(text, label):

    text = tf.expand_dims(text, -1)

    return vectorize_layer(text), label

# ENGLISH DATA
engData = preprocessData('language_data/eng_sentences.tsv', 0)
print("English data: \n", engData)

# SPANISH DATA
spaData = preprocessData('language_data/spa_sentences.tsv', 1)
print("Spanish data: \n", spaData)

# FRENCH DATA
fraData = preprocessData('language_data/fra_sentences.tsv', 2)
print("French data: \n", fraData)

# PORTUGUESE DATA
porData = preprocessData('language_data/por_sentences.tsv', 3)
print("Portuguese data: \n", porData)

# INDONESIAN DATA
indData = preprocessData('language_data/ind_sentences.tsv', 4)
print("Indonesian data: \n", indData)

# GERMAN DATA
deuData = preprocessData('language_data/deu_sentences.tsv', 5)
print("German data: \n", deuData)

# TURKISH DATA
turData = preprocessData('language_data/tur_sentences.tsv', 6)
print("Turkish data: \n", turData)

# VIETNAMESE DATA
vieData = preprocessData('language_data/vie_sentences.tsv', 7)
print("Vietnamese data: \n", vieData)

# ITALIAN DATA
itaData = preprocessData('language_data/ita_sentences.tsv', 8)
print("Italian data: \n", itaData)

# HUNGARIAN DATA
hunData = preprocessData('language_data/hun_sentences.tsv', 9)
print("Hungarian data: \n", hunData)

# combines into one dataframe
languageData = [engData, spaData, fraData, porData, indData, deuData, turData, vieData, itaData, hunData]
combinedData = pd.concat(languageData)

# splits dataframe into training (80%) and testing (20%) dataframes
trainingData, testingData = train_test_split(combinedData, test_size = 0.2)

# splits dataframes again into text (sentence) and label (language) dataframes
sentenceTrainingData = trainingData.copy()
sentenceTrainingData.pop("Language Index")

languageTrainingData = trainingData.copy()
languageTrainingData.pop("Sentence")

sentenceTestingData = testingData.copy()
sentenceTestingData.pop("Language Index")

languageTestingData = testingData.copy()
languageTestingData.pop("Sentence")

# exports training data for adapting vectorization layer in main
sentenceTrainingData.to_csv('sentenceTrainingData.tsv', sep='\t', index=False, encoding='utf-8')

# converts pandas dataframes to TensorFlow datasets
trainingData = tf.data.Dataset.from_tensor_slices((sentenceTrainingData, languageTrainingData))
testingData = tf.data.Dataset.from_tensor_slices((sentenceTestingData, languageTestingData))

# creates layer to vectorize text imputs
vectorize_layer = tf.keras.layers.TextVectorization(max_tokens = 2000, split = 'character', ngrams = 2)
sentenceTrainingData = trainingData.map(lambda x, y: x)
vectorize_layer.adapt(sentenceTrainingData)

# applies text vectorization layer
trainingData = trainingData.map(vectorize_text)
testingData = testingData.map(vectorize_text)

# optimizes performance
AUTOTUNE = tf.data.AUTOTUNE
trainingData = trainingData.cache().prefetch(buffer_size = AUTOTUNE)
testingData = testingData.cache().prefetch(buffer_size = AUTOTUNE)

# machine learning model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(2000, 128),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# compiles model
model.compile(
    optimizer = 'adam',
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)

# trains model
model.fit(trainingData, epochs = 2, batch_size = 10)

# checks loss and accuracy on testing data
loss, accuracy = model.evaluate(testingData)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# utilizes previous model to create a probability model
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# exports probability model for use in main
probability_model.save('language_probability_model.keras')