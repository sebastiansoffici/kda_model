#!/usr/bin/env python3

from keras import utils
from keras import layers
from keras import models
from keras.models import load_model
import numpy
import cv2
import random
import matplotlib
import datetime
import pandas
from matplotlib import pyplot
import tensorflow
from kda_data import DigitSample

class KDAModel(object):
  def __init__(self, num_samples = 1000000, image_sample = None):
    """KDAModel constructor. Initializes numpy arrays according to the number of samples with a given image_sample shape
      
        Args:
            num_samples: number of image samples
            image_sample: image sample of a K/D/A in the form (H,W,C), default is None for objects created to just use load model                           and show_predictions
        Returns: none
    """
    
    if image_sample is not None:
      self.images = numpy.zeros((num_samples, image_sample.shape[0], image_sample.shape[1], image_sample.shape[2]))
      
      self.tens_kills = numpy.zeros((num_samples, 10))
      self.kills = numpy.zeros((num_samples, 10))

      self.tens_deaths = numpy.zeros((num_samples, 10))
      self.deaths = numpy.zeros((num_samples, 10))

      self.tens_assists = numpy.zeros((num_samples, 10))
      self.assists = numpy.zeros((num_samples, 10))

      self.image_sample = image_sample
    
      self.num_samples = num_samples

  def to_one_hot(self,num):
    """Converts a one or two digit integer to one-hot representation
      
        Args:
            num: integer to be converted
        Returns:
            Array in the form [[tens_place], [ones_place]] where the index with a 1 corresponds to the digit
    """
    str_num = str(num)
    if len(str_num) < 2:
      str_num = '0' + str_num
    return tuple(utils.to_categorical(int(val), num_classes = 10) for val in str_num)

  def add_observation(self, i, image_sample, k, d, a):
    """Adds observations to later be evaluated by model
      
      Args:
          i: integer index
          image_sample: image sample K/D/A to be trained
          k: integer of kills correctly corresponding to the image_sample kills
          d: integer of deaths correctly corresponding to the image_sample  deaths
          a: integer of assits correctly corresponding to the image_sample assists
      Returns:
        none, operating on member arrays
    """
    self.images[i, :, :,:] = image_sample
    self.tens_kills[i,:], self.kills[i,:] = self.to_one_hot(k)
    self.tens_deaths[i,:], self.deaths[i,:] = self.to_one_hot(d)
    self.tens_assists[i,:], self.assists[i,:] = self.to_one_hot(a)
  

  """
  def add_block(self, convolutions = 4):
    pixels = layers.Input(shape = self.image_sample.shape)
    conv = layers.Conv2D(52, (3,3), activation = 'relu', padding = 'same')(pixels)
    for i in range(convolutions):
      conv = layers.Conv2D(52, (3,3), activation = 'relu', padding = 'same')(conv)

    return layers.MaxPooling2D(pool_size = (3,3), data_format = 'channels_last')(conv)
   """     
  
  def get_model(self, convolutions = 4):
    """Could use some help with a helpful description, ideally could use add_block to remove stuff from this method.
  
        Args:
          convolutions: number of times convolution is to be performed, default is 4
        Returns:
         Compiled model to be evaluated
    """
    pixels = layers.Input(shape = self.image_sample.shape)
    """
    pool = add_block(4)
    """
    conv = layers.Conv2D(52, (3,3), activation = 'relu', padding = 'same')(pixels)
    for i in range(convolutions):
      conv = layers.Conv2D(52, (3,3), activation = 'relu', padding = 'same')(conv)

    pool = layers.MaxPooling2D(pool_size = (3,3), data_format = 'channels_last')(conv)

    flat = layers.Flatten()(pool)
    x = layers.Dense(64, activation = 'relu')(flat)
    x = layers.Dense(64, activation = 'relu')(x)
              
    tens_kills = layers.Dense(10, activation = 'softmax', name = 'tens_kills')(x)
    kills = layers.Dense(10, activation = 'softmax', name = 'kills')(x)
              
    tens_deaths = layers.Dense(10, activation = 'softmax', name = 'tens_deaths')(x)
    deaths = layers.Dense(10, activation = 'softmax', name = 'deaths')(x)
    tens_assists = layers.Dense(10, activation = 'softmax', name = 'tens_assists')(x)
    assists = layers.Dense(10, activation = 'softmax', name = 'assists')(x)
    
    model = models.Model(inputs = pixels, outputs = [tens_kills, kills, tens_deaths, deaths, tens_assists, assists])
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
    return model
  
  def evaluate_model(self, model, num_epochs = 100):
    """Fit model with observations
    
        Args:
          model: keras model to be evaluated
          num_epochs: number of epochs to be run, default is 100
        Returns: none
    """
    model.fit(self.images, [self.tens_kills, self.kills, self.tens_deaths, self.deaths, self.tens_assists, self.assists], epochs = num_epochs)

  def save_model(self, model, filepath):
    """Saves keras model to a given filepath
      
        Args:
          model: given model that you want to save
          filepath: path to where you want the saved model
        Returns: none
    """
    model.save(filepath)
    del model

  def load_model(self, filepath):
    """Loads keras model from a given filepath
  
      Args:
        filepath: location of where the needed keras model lives
      Returns:
        Loaded keras model
    """
    return load_model(filepath)

  def show_predictions(self, model, digit_path, slash_path,num_predictions = 10):
    """Creates dataframe for true KDA versus predicted KDA for a given keras model

        Args:
            model: keras model to run predictions
            digit_path: filepath for digit samples
            slash_path: filepath for slash samples
            num_predictions: number of predictions wanted, default 10
        Returns: pandas dataframe
    """
    sample = DigitSample(digit_path, slash_path)
    data_frame = pandas.DataFrame(columns = ['True Kills', 'True Deaths', 'True Assists', 'Predicted Kills', 'Predicted Deaths', 'Predicted Assists', 'Model Kills Accuracy %', 'Model Deaths Accuracy %', 'Model Assists Accuracy %'])
    
    
    for i in range(num_predictions):
      k, d, a = sample.generate_random_kda()
      test_image = sample.get_kda(k,d,a)
      test_image = numpy.expand_dims(test_image, axis = 0)
      prediction = model.predict(test_image)
      data_frame = data_frame.append({'True Kills': k, 'True Deaths':d, 'True Assists':a, 'Predicted Kills': int(str(numpy.argmax(prediction[0]))+str(numpy.argmax(prediction[1]))),
                          'Predicted Deaths': int(str(numpy.argmax(prediction[2]))+str(numpy.argmax(prediction[3]))), 'Predicted Assists': int(str(numpy.argmax(prediction[4]))+str(numpy.argmax(prediction[5])))}, ignore_index=True)
    kill_count = 0
    death_count = 0
    assist_count = 0
    for index, row in data_frame.iterrows():
      if (row['True Kills'] == row['Predicted Kills']):
        kill_count+=1
      if (row['True Deaths'] == row['Predicted Deaths']):
        death_count+=1
      if (row['True Assists'] == row['Predicted Assists']):
        assist_count+=1
    kill_count = (kill_count/num_predictions)*100
    death_count = (death_count/num_predictions)*100
    assist_count = (assist_count/num_predictions)*100
    data_frame = data_frame.append({'Model Kills Accuracy %':kill_count, 'Model Deaths Accuracy %':death_count, 'Model Assists Accuracy %': assist_count}, ignore_index=True)
    return data_frame 
    
    

