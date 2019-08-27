import numpy as np
import os
import tensorflow as tf
import operations


# This class defines Memory architecture in DKVMN_bi
class DKVMN_Memory_bi():
    def __init__(self, memory_size, memory_state_dim, name):
        self.name = name
        print('%s initialized' % self.name)
        # Memory size : N
        self.memory_size = memory_size
        # Memory state dim : D_V or D_K
        self.memory_state_dim = memory_state_dim
        '''
			Key matrix or Value matrix
			Key matrix is used for calculating correlation weight(attention weight)
		'''

    def cor_weight(self, embedded, key_matrix, correctness, item_bias):
        '''
			embedded : [batch size, memory state dim(d_k)]
			item_bias : [batch size, 1]
			correctness : [batch size, 1]
			Key_matrix : [batch_size, memory size * memory state dim(d_k)]
			Correlation weight : w(i) = k * Key matrix(i)
			=> batch size * memory size
		'''
        # embedding_result : [batch size, memory size], each row contains each concept correlation weight for 1 question
        embedded_expand = tf.expand_dims(embedded, 1)
        embedding_result = tf.math.reduce_sum(tf.math.multiply(embedded_expand, key_matrix), axis=2, keepdims=False)
        # embedding_result = tf.matmul(embedded, tf.transpose(key_matrix))
        inner_prod = embedding_result + item_bias
        logits = tf.math.sigmoid(inner_prod)
        correlation_weight = tf.math.subtract(logits, tf.expand_dims(correctness, -1))
        #print('Correlation weight shape : %s' % (correlation_weight.get_shape()))
        return correlation_weight

    # Getting read content
    def read(self, value_matrix, correlation_weight, embedded, item_bias):
        '''
			embedded : [batch size, memory state dim(d_k)]
			item_bias : [batch size, 1]
			Value matrix : [batch size ,memory size ,memory state dim]
			Correlation weight : [batch size ,memory size], each element represents each concept embedding for 1 question
		'''
        # [batch size ,memory size]
        embedded_expand = tf.expand_dims(embedded, 1)
        inner_prod = tf.math.reduce_sum(tf.math.multiply(embedded_expand, value_matrix), axis=2, keep_dims=False)
        read_content = tf.math.add(inner_prod, item_bias)
       # print('Read content shape : %s' % (read_content.get_shape()))
        return read_content

    def write(self, value_matrix, correlation_weight, q_embedded, reuse=False):
        '''
			Value matrix : [batch size, memory size, memory state dim(d_k)]
			Correlation weight : [batch size, memory size]
			qa_embedded : (q) pair embedded, [batch size, memory state dim(d_v)]
		'''
        update = operations.linear2(correlation_weight, name=self.name + '/Update_vector', reuse=reuse)
        update_tiled = tf.tile(tf.expand_dims(update, -1), tf.stack([1, 1, self.memory_state_dim]))
        q_embedded_expand = tf.expand_dims(q_embedded, 1)
        new_memory = tf.math.subtract(value_matrix, tf.math.multiply(update_tiled, q_embedded_expand))
        # [batch size, memory size, memory value staet dim]
        # print('Memory shape : %s' % (new_memory.get_shape()))
        return new_memory


# This class construct key matrix and value matrix
class DKVMN_bi():
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key, init_memory_value,
                 name='DKVMN_bi'):
        print('Initializing memory..')
        self.name = name
        self.memory_size = memory_size
        # self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        # self.key = DKVMN_Memory(self.memory_size, self.memory_key_state_dim, name=self.name+'_key_matrix')
        self.value = DKVMN_Memory_bi(self.memory_size, self.memory_value_state_dim, name=self.name + '_value_matrix')

        # self.memory_key = init_memory_key
        self.memory_value = init_memory_value

    def attention(self, q_embedded, q_bias, correctness):
        correlation_weight = self.value.cor_weight(embedded=q_embedded, key_matrix=self.memory_value,
                                                   correctness=correctness, item_bias=q_bias)
        return correlation_weight

    def read(self, c_weight, q_embedded, q_bias):
        read_content = self.value.read(value_matrix=self.memory_value, correlation_weight=c_weight, embedded=q_embedded,
                                       item_bias=q_bias)
        return read_content

    def write(self, c_weight, q_embedded, reuse):
        self.memory_value = self.value.write(value_matrix=self.memory_value, correlation_weight=c_weight,
                                             q_embedded=q_embedded, reuse=reuse)
        return self.memory_value
