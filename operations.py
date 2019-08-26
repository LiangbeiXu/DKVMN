import numpy as np
import tensorflow as tf
import os

# x : input(batch * state_dim)
def linear(x, state_dim, name='linear', reuse=True):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()

		weight = tf.get_variable('weight', [x.get_shape()[-1], state_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
		bias = tf.get_variable('bias', [state_dim], initializer=tf.constant_initializer(0))
		weighted_sum = tf.matmul(x, weight) + bias
		return weighted_sum
		

def linear2(x, name='linear2', reuse=True):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()
		learning_rate = tf.get_variable('learning_rate', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		scaled_x = tf.math.multiply(tf.tile(tf.expand_dims(learning_rate, 0), tf.shape(x)), x)
		return scaled_x
