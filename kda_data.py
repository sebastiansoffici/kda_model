#!/usr/bin/env python3

import cv2
from matplotlib import pyplot
import numpy
import random
import os
from PIL import Image

class DigitSample(object):
  def __init__(self, digit_path, slash_path):
    """Initializes file path when an instance of a DigitSample is created
      
        Args:
          digit_path: string that contains the filepath of the digit images
          slash_path: string that contains the filepath of the slash images
    """
    self.digit_fp = digit_path
    self.slash_fp = slash_path

  def randomly_sample_digit(self, i):
    """Randomly samples digit from selection of images for a given integer

      Args:
          i: integer that the sample is needed for
      Returns:
          Random image sample of given integer
    """
    folder = self.digit_fp
    folder = os.path.join(folder, str(i))
    a = random.choice(os.listdir(folder))
    file_name = os.path.join(folder, a)
    digit = cv2.imread(file_name, cv2.IMREAD_COLOR)
    return digit
 
  def generate_random_kda(self):
    """Generates a random kda to be used in training
      
      Args: none
      Returns:
          A list in the form [kills, deaths, assists]
    """
    kda = [random.randrange(100), random.randrange(100), random.randrange(100)]
    return kda

  def get_slash(self):
    """Randomly chooses a slash from sample images 
      
      Args: none
      Returns:
          Random image sample of slash
    """
    a = random.choice(os.listdir(self.slash_fp))
    file_name = os.path.join(self.slash_fp, a)
    slash = cv2.imread(file_name, cv2.IMREAD_COLOR)
    return slash
 
  def transform(self, image):
    """Randomly crops and pads an image
    
      Args:
          image: an image of an integer, sampled using randomly_sample_digit
      Returns:
          Transformed image as a numpy array in the form (Height, Width, Channels)
    """

    crop_left = random.randrange(2)
    crop_right = image.shape[1] - random.randrange(2)
    cropped = image[:, crop_left:crop_right, :]
    space_left = numpy.zeros((image.shape[0], random.randrange(2), image.shape[2]), dtype='uint8')
    space_right = numpy.zeros((image.shape[0], random.randrange(2), image.shape[2]), dtype='uint8')
    return numpy.concatenate([space_left, cropped, space_right], axis = 1)

  def get_digits(self, number):
    """Transforms each digit of a number individually and then concatenates them
        
        Args:
            number: integer that corresponds to a kill, death, or assist
        Returns:
            Transformed and concanetated image as a numpy array in the form (Height, Width, Channels)
    """
    args = [self.transform(self.randomly_sample_digit(i)) for i in str(number)]
    return numpy.concatenate([self.transform(self.randomly_sample_digit(i)) for i in str(number)], axis = 1)

  def get_kda_cropped(self, k, d, a):
    """Creates a tightly cropped K/D/A image
        
        Args: 
            k: integer corresponding to kills
            d: integer corresponding to deaths
            a: integer corresponding to assists
        Returns:
            A K/D/A image as a numpy array in the form (Height, Width, Channels)
    """
    args = [self.get_digits(k), self.get_slash(), self.get_digits(d), self.get_slash(), self.get_digits(a)]
    return numpy.concatenate(args, axis = 1)

  def get_kda(self, k, d, a):
    """Creates a K/D/A image embedded in a canvas with the target prediction zone size
    
        Args:
            k: integer corresponding to kills
            d: integer corresponding to deaths
            aL integer corresponding to assists
        Returns:
            A K/D/A image within target prediction zone size as a numpy array in the form (Height, Width, Channels)
    """
    canvas = numpy.zeros((9, 34 + 2 * 10, 3), dtype = 'uint8')
    cropped = self.get_kda_cropped(k, d, a)
    width = cropped.shape[1]
    start_range = canvas.shape[1] - width
    if start_range < 1:
      start_range = 1
    start = random.randrange(start_range)
    end_canvas = min(canvas.shape[1], start+width)
    canvas[:, start:end_canvas, :] = cropped[:, 0:(end_canvas - start), :]
    return canvas
  

