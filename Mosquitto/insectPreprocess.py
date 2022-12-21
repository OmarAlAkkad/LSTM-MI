# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 18:29:03 2022

@author: omars
"""
import numpy as np
import csv
import pickle
from sklearn.model_selection import train_test_split

def preprocess_dataset():
    # load all data
    f = open('InsectWingbeatSound_TEST', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    page = []
    for line in rdr:
        page.append(line)
    f.close()


    f = open('InsectWingbeatSound_TRAIN', 'r', encoding='utf-8')
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
        if int(line[0]) == 11:
            line[0] = 0
        labels.append(int(line[0]))

    return inputs, labels

if __name__ == '__main__':
    dataset = preprocess_dataset()

    inputs, labels = create_inputs_labels(dataset)

    target_model_data, shadow_model_data, target_model_labels, shadow_model_labels = train_test_split(inputs, labels,stratify=labels, test_size=0.50, random_state=42)

    pickle.dump(target_model_data, open('insect_target_data.p', 'wb'))
    pickle.dump(target_model_labels, open('insect_target_labels.p', 'wb'))
    pickle.dump(shadow_model_data, open('insect_shadow_data.p', 'wb'))
    pickle.dump(shadow_model_labels, open('insect_shadow_labels.p', 'wb'))



