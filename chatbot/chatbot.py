import numpy as np
import time
from preprocess import read_txt, model_inputs
from seq2seq import seq2seq_model, apply_padding
from preprocess_answers_and_questions import preprocess_data

import tensorflow as tf

lines = read_txt('movie_lines.txt')
conversations = read_txt('movie_conversations.txt')

epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5


tf.reset_default_graph()
session = tf.InteractiveSession()       # Defining a session


ans_words_to_int, ques_words_to_int, sort_clean_ques, sort_clean_ans = preprocess_data(lines, conversations)
inputs, targets, learn_rate, keep_prob = model_inputs()
sequence_length = tf.placeholder_with_default(25, None, name='sequence_length')
input_shape = tf.shape(inputs)

training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(ans_words_to_int),
                                                       len(ques_words_to_int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       ques_words_to_int)

with tf.name_scope('optimization'):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable)
                         for grad_tensor, grad_variable in gradients
                         if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

training_validation_split = int(len(sort_clean_ques) * 0.15)

training_questions = sort_clean_ques[training_validation_split:]
training_answers = sort_clean_ans[training_validation_split:]

validation_questions = sort_clean_ques[:training_validation_split]
validation_answers = sort_clean_ans[:training_validation_split]
