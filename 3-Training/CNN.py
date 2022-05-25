import matplotlib
from matplotlib import pyplot as plt
import keras.api._v2.keras as keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# creating input layer
tf.keras.Input(shape=[1], dtype=tf.float32)
normalizer = tf.keras.layers.Normalization(axis=-1)
