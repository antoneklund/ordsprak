'''
Swedish ordspråk generator.
Anton Eklund

Imports and applies neural network onto a set of Swedish sayings.
Try to create new sayings from that data.
'''

print("Starting swedish ordspråk generator.")
print("Importing modules")

import random as r
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt


ORDSPRAK_PATH = "listaordsprak.xlsx"
ORDSPRAK_COL_NAME = "Ordspråk"
SAVE_PATH = "C:\Code\python\machinelearning\ordsprak" #On my personal copmuter


WINDOW_SIZE = 3
EMBEDDING_SIZE = 10
LEARNING_RATE = 3
EPOCHS = 100
MAX_LENGTH_GEN_SENTENCE = 10

def import_data():
    list_ordsprak = pd.read_excel(ORDSPRAK_PATH)
    #print(len(list_ordsprak))
    filtered_list = pd.DataFrame()
    for i, ordsprak in list_ordsprak.iterrows():
        filtered = filter_sentence(ordsprak[ORDSPRAK_COL_NAME]).lower()
        filtered_list = filtered_list.append(pd.DataFrame([filtered]), ignore_index=True)
    return filtered_list


def filter_sentence(sentence):
    filtered = ""
    for char in sentence:
        if char == '.':
            break
        else:
            filtered = filtered + char
    return filtered



def create_dictionary(list_of_sentences):
    words = []
    for i, sentence_ser in list_of_sentences.iterrows():
        sentence = sentence_ser.tolist()
        for word in sentence[0].split():
            if word not in words:
                words.append(word)
    # words.append('oob')
    int2word = {}
    word2int = {}
    for i, word in enumerate(words):
        word2int[word] = i
        int2word[i] = word

    return int2word, word2int, words

def generate_batch(window_size, dataframe, word2int):
    batch = []
    for i, sentence_ser in dataframe.iterrows():
        sentence = sentence_ser.tolist()
        word_vec = []
        sentence_list = sentence[0].split()
        sentence_length = len(sentence_list)
        # print(sentence_list)
        # print(sentence_length)
        for i, word in enumerate(sentence_list):
            for neighbour in sentence_list[i : min(i + WINDOW_SIZE, sentence_length + 1)]:
                if neighbour != word:
                    batch.append([word2int[word], word2int[neighbour]])


    #     for i, word in enumerate(sentence_list):
    #         for neighbour in sentence_list[max(i - WINDOW_SIZE, 0) : min(i + WINDOW_SIZE, sentence_length + 1)]:
    #             if neighbour != word:
    #                 batch2.append([word, neighbour])
    #
    # print(batch2)
    return batch

def create_training_vectors(batch, dict_size):
    x_train = [] # input word
    y_train = [] # output word
    for item in batch:
        x_train.append(to_one_hot(item[0], dict_size))
        y_train.append(to_one_hot(item[1], dict_size))
        # print(x_train)
        # print(y_train)
        # convert them to numpy arrays
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    assert not np.any(np.isnan(x_train))
    assert not np.any(np.isnan(y_train))
    return x_train, y_train

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

def generate_candidates(session, word, tf_variables, word2int, int2word):
    prediction = tf_variables[-1]
    x = tf_variables[0]
    y_label = tf_variables[1]
    x_train = tf_variables[2]
    y_train = tf_variables[3]
    result = session.run(prediction, feed_dict={x:x_train, y_label: y_train })
    next_word = []
    # for result in results:
    for i, value in enumerate(result[word2int[word]]):
        if value > 0.02:
            next_word.append(int2word[i])
    return next_word


def create_ordsprak(session, word, tf_variables, word2int, int2word):
    new_ordsprak = word
    nr_words= 0
    while (nr_words < MAX_LENGTH_GEN_SENTENCE):
        candidates = generate_candidates(session, word, tf_variables, word2int, int2word)
        if len(candidates) == 0:
            candidates.append("och")
        next_word = r.choice(candidates)
        print(candidates)
        new_ordsprak = new_ordsprak + " " + next_word
        word = next_word
        print(word)
        nr_words += 1
    return new_ordsprak



#While loop to keep program running and interact with user
def interface(session, vectors, tf_variables, word2int, int2word, words):
    option = input("Enter option (type \"exit\" to exit): ")
    while str(option).lower() != 'exit':
        if option not in words:
            print("Not a valid word")
        else:
            print(create_ordsprak(session, option.lower(), tf_variables, word2int, int2word))
        option = input("Enter option: ")


def main():
    df = import_data()
    int2word, word2int, words = create_dictionary(df)
    batch = generate_batch(WINDOW_SIZE, df, word2int)
    dict_size = len(int2word)
    x_train, y_train = create_training_vectors(batch, dict_size)

    # making placeholders for x_train and y_train
    x = tf.placeholder(tf.float32, shape=(None, dict_size))
    y_label = tf.placeholder(tf.float32, shape=(None, dict_size))

    #Create variables
    W1 = tf.Variable(tf.random_normal([dict_size, EMBEDDING_SIZE]))
    b1 = tf.Variable(tf.random_normal([EMBEDDING_SIZE])) #bias
    hidden_representation = tf.add(tf.matmul(x,W1), b1)

    W2 = tf.Variable(tf.random_normal([EMBEDDING_SIZE, dict_size]))
    b2 = tf.Variable(tf.random_normal([dict_size]))
    prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    #Loss function
    loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

    #Step function
    step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    #Train model for EPOCHS number of epochs
    for _ in range(EPOCHS):
        session.run(step, feed_dict={x: x_train, y_label: y_train})
        print('loss is : ', session.run(loss, feed_dict={x: x_train, y_label: y_train}))

    vectors = session.run(W1 + b1)

    #Save model
    saver = tf.train.Saver()
    save_path = saver.save(session, SAVE_PATH + "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

    #Enter interface to create new sayings
    tf_variables = [x, y_label, x_train, y_train, W1, b1, W2, b2, hidden_representation, prediction]
    interface(session, vectors, tf_variables, word2int, int2word, words)
if __name__ == '__main__':
    main()
