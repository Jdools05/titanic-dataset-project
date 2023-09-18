import csv
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

PATH_TO_TEST = './data/test.csv'
PATH_TO_TRAIN = './data/train.csv'

def get_data_cleaned(file):
    data = pd.read_csv(file)

    print(data.head())

    # if has key survived, then this is the training data
    target = data.pop('Survived') if 'Survived' in data else None
    data.pop('PassengerId')
    data.pop('Name')
    data.pop('Ticket')

    # convert binary categorical data to 0/1 (sex)
    data['Sex'] = data['Sex'].map({'male':0, 'female':1})

    # convert categorical data to one-hot encoding (embarked)
    data = pd.get_dummies(data, columns=['Embarked'])

    # convert cabin to 0 if null, otherwise convert the room numbers to unique integers
    # clean up so that each entry has a single room number
    data['Cabin'] = data['Cabin'].str.split(' ').str[0]
    data['Cabin'] = data['Cabin'].apply(lambda x: 0 if pd.isnull(x) else int(x, 36))

    # convert fare to 0 if null
    data['Fare'] = data['Fare'].apply(lambda x: 0 if pd.isnull(x) else x)

    #make sure all data is numeric
    data = data.astype('float64')

    # clean any other null values
    data = data.fillna(0)

    # normalize the age, fare, pclass, and cabin
    data['Age'] = data['Age'] / data['Age'].max()
    data['Fare'] = data['Fare'] / data['Fare'].max()
    data['Pclass'] = data['Pclass'] / data['Pclass'].max()
    data['Cabin'] = data['Cabin'] / data['Cabin'].max()


    print(data.head())

    return data, target

def main():
    # with open(PATH_TO_TRAIN, 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         print(row)
    
    # people = []

    # with open(PATH_TO_TRAIN, 'r') as f:
    #     reader = csv.reader(f, delimiter=',', quotechar='"')
    #     for row in reader:
    #         if row[0] == 'PassengerId':
    #             continue
    #         people.append(Person(*row))

    data = None
    target = None

    with open(PATH_TO_TRAIN, 'r') as f:
        data, target = get_data_cleaned(f)

        print(data.head())
        print(target.head())


    # assert that the data has no null values or NaN values
    assert data.isnull().sum().sum() == 0
    assert data.isna().sum().sum() == 0


    

    # create a model to train on the data
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    model.fit(data, target, epochs=128, batch_size=32)

    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir="LR", expand_nested=True, dpi=96)

    # test the model on the test data
    with open(PATH_TO_TEST, 'r') as f:
        data, _ = get_data_cleaned(f)
    
    predictions = model.predict(data).flatten()

    # make the values survived to 0 or 1
    predictions = np.round(predictions).astype(bool)

    # get the accuracy of the model
    # accuracy = model.evaluate(data, target)

    print(predictions)

    

    

        
        
    


if __name__ == '__main__':
    main()