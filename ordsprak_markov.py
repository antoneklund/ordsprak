'''
Swedish ordspråk generator.
Anton Eklund

Imports and applies markov chain onto a set of Swedish sayings.
Try to create new sayings from that data.
'''

print("Starting swedish ordspråk generator.")
print("Importing modules")

import random as r
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

BAD_CHARS = [',', '<', '>', '!', '?', '-','<', ':',';','*', '(', ')']

ORDSPRAK_PATH = "listaordsprak.xlsx"
ORDSPRAK_COL_NAME = "Ordspråk"
SAVE_PATH = "C:\Code\python\machinelearning\ordsprak" #On my personal copmuter


WINDOW_SIZE = 3
EMBEDDING_SIZE = 10
LEARNING_RATE = 3
EPOCHS = 40
MAX_LENGTH_GEN_SENTENCE = 10
AMOUNT_ORDSPRAK = 10


def has_digit(word):
    for char in word:
        if char.isdigit():
            return True
    return False


def import_data():
    list_ordsprak = pd.read_excel(ORDSPRAK_PATH)
    filtered_list = pd.DataFrame()
    for i, ordsprak in list_ordsprak.iterrows():
        filtered = filter_sentence(ordsprak[ORDSPRAK_COL_NAME]).lower()
        filtered_list = filtered_list.append(pd.DataFrame([filtered]), ignore_index=True)
    return filtered_list


def filter_sentence(sentence):
    filtered = ""
    for char in sentence:
        if char not in BAD_CHARS:
            if char == '.':
                break
            else:
                filtered = filtered + char
    return filtered



def create_dictionary(list_of_sentences):
    print("Creating dictionary")
    words = []
    for i, sentence_ser in list_of_sentences.iterrows():
        sentence = sentence_ser.tolist()
        for word in sentence[0].split():
            if word not in words:
                words.append(word)
    int2word = {}
    word2int = {}
    for i, word in enumerate(words):
        word2int[word] = i
        int2word[i] = word

    return int2word, word2int, words


def create_current_next_dfs(sentences):
    print("Create current-next database")
# def convert_sentences_to_words(sentences):
    current_next_df = pd.DataFrame(columns = ["current", "next", "count"])
    end_word_df = pd.DataFrame(columns = ["end"])
    end_word = ""
    for i, sentence_ser in sentences.iterrows():
        sentence = sentence_ser.tolist()[0].split()
        sentence_length = len(sentence)
        for j, current in enumerate(sentence):
            if j != sentence_length-1:
                next = sentence[min(j+1, sentence_length+1)]
                match = current_next_df["current"].isin([current])
                row = pd.DataFrame([[current, next, 1.0]], columns= ["current", "next", "count"])
                current_next_df = current_next_df.append(row, ignore_index=True)

            else:
                next = "END"
                match = current_next_df["current"].isin([current])
                row = pd.DataFrame([[current, next, 1.0]], columns= ["current", "next", "count"])
                current_next_df = current_next_df.append(row, ignore_index=True)

                if not (current in end_word_df.values):
                    end_word_df = end_word_df.append(pd.DataFrame([current], columns=["end"]), ignore_index=True)

                # print(end_word_df)
    return current_next_df, end_word_df

'''
(current, [(next, prob)])

'''
def calculate_prbability(df_old):
    print("Calculate probabilities")
    probability_table = []
    words = []
    df = pd.DataFrame(columns=["current", "next", "count", "freq"])
    df = df_old
    for _, row in df.iterrows():
        word = row["current"]
        if word not in words:
            matches = df[(row["current"] == df["current"])]
            count_sum = matches["count"].sum()
            indexes = matches.index.tolist()
            for i in indexes:
                count = 0.0 + df["count"][i]
                if count < 1:
                    value_to_set = 1
                else:
                    value_to_set = count/count_sum
                    df.at[i, "freq"] = value_to_set
            words.append(word)
    for word in words:
        next_words = []
        matches = df[(df["current"] == word)]
        for i, match in matches.iterrows():
            next = match["next"]
            freq = match["freq"]
            next_words.append((next, freq))
        probability_table.append((word, next_words))
    return probability_table, words

def get_index(word, l):
    for i, item in enumerate(l):
        if item[0] == word:
            return i

def create_ordsprak(pt, word):
    ordsprak = [word]
    while(len(ordsprak) < 20):
        options = pt[get_index(word, pt)][1]
        probabilities = []
        for o in options:
            probabilities.append(o[1])
        probabilities = np.cumsum(probabilities)
        choice = r.random()
        for i, p in enumerate(probabilities):
            if choice < p:
                if options[i][0] == "END":
                    ordsprak.append('.')
                    return ordsprak
                ordsprak.append(" " + options[i][0])
                word = options[i][0]

                break
    return ordsprak



def interface(pt, words):
    option = input("Enter option (type \"exit\" to exit): ").lower()
    while str(option).lower() != 'exit':
        if option not in words:
            print("Not a valid word")
        else:
            for k in range(AMOUNT_ORDSPRAK):
                print("".join(create_ordsprak(pt, option)))
        option = input("Enter option: ").lower()



def main():
    df = import_data()
    int2word, word2int, words = create_dictionary(df)
    current_next_df, end_word_df = create_current_next_dfs(df)
    current_next_df['count']= current_next_df.groupby(by=['current','next'])['current','next'].transform('count').copy()
    current_next_df = current_next_df.drop_duplicates()
    probability_table, words = calculate_prbability(current_next_df)
    interface(probability_table, words)


if __name__ == "__main__":
    main()
