import tensorflow as tf


class s2s_model:
    def __init__():
        self.build_embedding()
        self.build_encoding()
        self.build_decoding()
        self.build_loss()
    def build_embedding():

    def build_encoding():
        # Build RNN cell
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        # Run Dynamic RNN
        #   encoder_outpus: [max_time, batch_size, num_units]
        #   encoder_state: [batch_size, num_units]
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, encoder_emb_inp,
            sequence_length=source_sequence_length, time_major=True)

    def build_decoding():
        # Build RNN cell
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp, decoder_lengths, time_major=True)
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, encoder_state,
            output_layer=projection_layer)
        # Dynamic decoding
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
        logits = outputs.rnn_output
    def build_loss():