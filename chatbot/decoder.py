import tensorflow as tf


def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input,
                        sequence_length, decoding_scope, output_function,
                        keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_fn, attention_construct_fn =
        tf.contrib.seq2seq.prepare_attention(attention_states,
                                             attention_option='bahdanau',
                                             num_units=decoder_cell.output_size)
    dynamic_fn_train = tf.contrib.seq2seq.attention_decoder_fn_train(
                                                      encoder_state[0],
                                                      attention_keys,
                                                      attention_values,
                                                      attention_score_fn,
                                                      attention_construct_fn,
                                                      name='attn_dec_train')
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                  dynamic_fn_train,
                                                                  decoder_embedded_input,
                                                                  sequence_length,
                                                                  scope=decoding_scope)
    decoder_output_droppout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_droppout)


def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id,
                    eos_id, maximum_length, num_words, sequence_length, decoding_scope,
                    output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_fn, attention_construct_fn =
        tf.contrib.seq2seq.prepare_attention(attention_states,
                                             attention_option='bahdanau',
                                             num_units=decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(
                                                          output_function,
                                                          encoder_state[0],
                                                          attention_keys,
                                                          attention_values,
                                                          attention_score_fn,
                                                          attention_construct_fn,
                                                          sos_id,
                                                          eos_id,
                                                          maximum_length,
                                                          num_words,
                                                          name='attn_dec_inf')
    test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                    test_decoder_function,
                                                                    scope=decoding_scope)
    return test_predictions


def decoder_rnn(decoder_embedded_input, decoder_embeddedings_matrix, encoder_state,
                num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob,
                batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_fn = lambda x: tf.contrib.layers.fully_connected(x,
                                                                num_words,
                                                                None,
                                                                scope=decoding_scope,
                                                                weights_initializer=weights,
                                                                biases_initializer=biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_fn,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_celll,
                                           decoder_embeddings_matrix,
                                           word2int[' <SOS> '],
                                           word2int[' <EOS> '],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_fn,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions