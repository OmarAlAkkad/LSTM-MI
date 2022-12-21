    # -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 17:59:06 2022

@author: omars
"""
import numpy as np
import csv
import pickle
from sklearn.model_selection import train_test_split

def preprocess_dataset():
    # load all data
    f = open('CBF_TRAIN.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    page = []
    for line in rdr:
        page.append(line)
    f.close()


    f = open('CBF_TEST.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        page.append(line)
    f.close()


    return page

def create_inputs_labels(dataset):
    inputs = []
    labels = []
    for line in dataset:
        inputs.append(list(map(float, line[1:])))
        labels.append(int(line[0]))

    return inputs, labels

if __name__ == '__main__':
    dataset = preprocess_dataset()

    inputs, labels = create_inputs_labels(dataset)

    actual_model_data, shadow_model_data, actual_model_labels, shadow_model_labels = train_test_split(inputs, labels, test_size=0.50, random_state=42)

    pickle.dump(actual_model_data, open('CBF_actual_data.p', 'wb'))
    pickle.dump(actual_model_labels, open('CBF_actual_labels.p', 'wb'))
    pickle.dump(shadow_model_data, open('CBF_shadow_data.p', 'wb'))
    pickle.dump(shadow_model_labels, open('CBF_shadow_labels.p', 'wb'))



