import time
from preprocess import read_txt, model_inputs
from seq2seq import seq2seq_model, split_into_batches
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
inputs, targets, lr, keep_prob = model_inputs()
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

batch_index_check_training_loss = 100
batch_index_check_validation_loss = (len(training_questions))
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())


for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) \
     in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping,
                                                    loss_error],
                                                   {inputs: padded_questions_in_batch,
                                                    targets: padded_answers_in_batch,
                                                    lr: learning_rate,
                                                    sequence_length: padded_answers_in_batch.shape[1],
                                                    keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            tr_loss_er = total_training_loss_error / batch_index_check_training_loss
            time_100_batches = batch_time * batch_index_check_training_loss
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, \
             Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                epochs,
                                                                batch_index,
                                                                len(training_questions),
                                                                tr_loss_er,
                                                                int(time_100_batches)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) \
                    in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                _, batch_validation_loss_error = session.run(
                    loss_error,
                    {inputs: padded_questions_in_batch,
                     targets: padded_answers_in_batch,
                     lr: learning_rate,
                     sequence_length: padded_answers_in_batch.shape[1],
                     keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (
                                            len(validation_questions) / batch_size)
            print("Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds"
                  .format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print("I speak better now!!!")
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I don't speak better. I need to practice more!!!")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("Unfortunatelly, I cannot speak better anymore. This is the best I can do!")
        break

print("Game Over")
