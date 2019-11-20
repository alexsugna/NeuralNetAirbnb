#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the class NeuralNet which includes many useful functions
from tensorflow and sklearn for training and evaluating models.

Alex Angus, John Dale

November 13, 2019
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt 
from keras import regularizers

class NeuralNet:
    """
    Neural Network model implemented with tensorflow and evaluation methods
    implemented with sklearn.        
    """
    def __init__(self):
        self.model = Sequential()
    
    def add_first_layer(self, input_shape, num_features, activation_function):
        """
        Add the first layer to the neural net.
        
        num_features: an int specifying the number of features, or nodes
        activation function: string specifying the type of activation function
        """
        self.model.add(Dense(input_shape, input_dim=num_features, 
                             kernel_initializer='normal', 
                             activation=activation_function))
                             
    def add_layer(self, num_nodes, activation_function, regularization=None):
        """
        Add a hidden layer to the neural net.
        
        num_nodes: int specifying the number of hidden nodes
        activation_function: string specifying the type of activation function
        """
        if regularization != None:
            regularization = regularizers.l1(0.01)
        self.model.add(Dense(num_nodes, kernel_initializer='normal', activation=activation_function,
                             kernel_regularizer=regularization))
    
    def add_last_layer(self):
        """
        Add the last layer of the NN. We are doing regression, so there will 
        be only one output.
        """
        self.model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        
        
        
    def compile(self, loss_function, optimizer, epochs, batch_size, 
                verbosity):
        """
        Compile the model for training and build an estimator

        loss_function: string specifying the loss function that will be used
        optimizer: string specifying the optimization method
        epochs: int specifying the number of epochs
        batch_size: int specifying the batch size
        verbosity: int specify the level of textual feedback
        validation_split: percentage of data that will be used for validation                 
        """
        self.batch_size = batch_size
        self.model.compile(loss=loss_function, optimizer=optimizer, metrics=['mean_squared_error'])
        self.estimator = KerasRegressor(build_fn=self.get_model, epochs=epochs, 
                                        batch_size=batch_size, 
                                        verbose=verbosity)
        print(self.model.summary())
                                        
    def get_model(self):
        """
        Returns the NeuralNet model, because this is the only way it keras will work.
        """
        return self.model

        
    def train(self, X, y, epochs, batch_size, validation_split):
        """
        Train the neural network. 
        
        params:
            X: set of features
            y: set of targets
            validation_split: a float from 0 to 1 specifying the portion of the set to use as
                              validation
        
        returns:
            a history object containing the training and validation loss at each epoch
        """
        return self.estimator.fit(X, y, epochs=epochs, batch_size=batch_size, 
                                  validation_split=validation_split, shuffle=True)
        
        
    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluate the neural network.
        
        """
        self.test_predictions = self.model.predict(X_test)
        self.train_predictions = self.model.predict(X_train)
        
        print("Training MSE:", round(mean_squared_error(y_train, self.train_predictions),4))
        print("Validation MSE:", round(mean_squared_error(y_test, self.test_predictions),4))
        print("\nTraining r2:", round(r2_score(y_train, self.train_predictions),4))
        print("Validation r2:", round(r2_score(y_test, self.test_predictions),4))

        self.results = self.model.history.history
        plt.plot(list(range(1,len(self.results['loss'])+1)), self.results['loss'][0:], label='Train')
        plt.plot(list(range(1,len(self.results['val_loss'])+1)), self.results['val_loss'][0:], label='Test', color='green')
        plt.legend()
        plt.title('Training and test loss at each epoch', fontsize=14)
        plt.show()
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Predictions vs. Targets', fontsize=14, y=1)
        plt.subplots_adjust(top=0.93, wspace=0)
    
        ax1.scatter(y_test, self.test_predictions, s=2, alpha=0.7)
        ax1.plot(list(range(2,8)), list(range(2,8)), color='black', linestyle='--')
        ax1.set_title('Test')
        ax1.set_xlabel('Targets')
        ax1.set_ylabel('Predictions')
    
        ax2.scatter(y_train, self.train_predictions, s=2, alpha=0.7)
        ax2.plot(list(range(2,8)), list(range(2,8)), color='black', linestyle='--')
        ax2.set_title('Training')
        ax2.set_xlabel('Targets')
        ax2.set_ylabel('')
        ax2.set_yticklabels(labels='')
        plt.show()
        """
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
