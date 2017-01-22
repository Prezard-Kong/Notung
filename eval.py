from __future__ import unicode_literals
from __future__ import print_function
from model import Vgg16
from model import LSTM
import tensorflow as tf
import numpy as np
import pickle
import util


class Evaluator(object):

    def __init__(self, seq_length=16, beam_size=2):
        self.__seq_length = seq_length
        self.__beam_size = beam_size
        self.__vgg = Vgg16()
        self.__lstm = LSTM(batch_size=self.__beam_size)
        self.__vocab = pickle.load(open('./data/vocab.pickle'))
        self.__vocab_size = len(self.__vocab)
        self.__vgg_sess = tf.Session(graph=self.__vgg.graph)
        self.__lstm_sess = tf.Session(graph=self.__lstm.graph)
        saver = tf.train.Saver(self.__vgg.varlist)
        saver.restore(self.__vgg_sess, './data/cnn/cnn.ckpt')
        saver = tf.train.Saver(self.__lstm.varlist)
        saver.restore(self.__lstm_sess, './data/lstm/lstm.ckpt')

    def __seq_postprocess(self, seq):
        tmp = []
        for s in seq:
            if s >= self.__vocab_size-1:
                break
            tmp.append(self.__vocab[s])
        self.caption = ' '.join(tmp)

    def __get_sequence_beam(self):
        beam_seq = (self.__vocab_size-1)*np.ones((self.__seq_length, self.__beam_size), dtype=np.int32)
        probs_sum = np.zeros(self.__beam_size)
        flags = [True]*self.__beam_size
        beam_result = []
        feed_dict = {self.__lstm.data: np.tile(self.features, (self.__beam_size, 1)),
                     self.__lstm.word_input: np.ones(self.__beam_size),
                     self.__lstm.prev_c: np.zeros((self.__beam_size, 768)),
                     self.__lstm.prev_h: np.zeros((self.__beam_size, 768)),
                     self.__lstm.flag: 1.0}
        fetches = [self.__lstm.prob, self.__lstm.next_c, self.__lstm.next_h]
        c = np.zeros((self.__beam_size, 768))
        h = np.zeros((self.__beam_size, 768))
        for i in range(self.__seq_length+2):
            if i == 1:
                feed_dict[self.__lstm.word_input] = self.__vocab_size*np.ones(self.__beam_size)
                feed_dict[self.__lstm.flag] = 0.0
            elif i > 1:
                rows = self.__beam_size if i > 2 else 1
                candicates = []
                for row in range(rows):
                    for col in range(prob.shape[1]):
                        candicate_prob = probs_sum[row] + prob[row, col]
                        candicates.append((candicate_prob, row, col, prob[row, col]))
                candicates.sort(key=lambda cp: cp[0], reverse=True)

                new_c = np.copy(c)
                new_h = np.copy(h)
                if i > 2:
                    beam_seq_prev = np.copy(beam_seq[:i-2, :])
                for j in range(self.__beam_size):
                    candi = candicates[j]
                    if i > 2:
                        beam_seq[:i-2, j] = beam_seq_prev[:i-2, candi[1]]
                    c[j, :] = np.copy(new_c[candi[1], :])
                    h[j, :] = np.copy(new_h[candi[1], :])
                    beam_seq[i-2, j] = candi[2]
                    probs_sum[j] = candi[0]
                    if (candi[2] >= self.__vocab_size-1 or i == self.__seq_length-1) and flags[j] is True:
                        beam_result.append((candi[0], np.copy(beam_seq[:, j])))
                        flags[j] = False
                    feed_dict[self.__lstm.word_input] = beam_seq[i-2, :]
            feed_dict[self.__lstm.prev_c] = c
            feed_dict[self.__lstm.prev_h] = h
            prob, c, h = self.__lstm_sess.run(fetches=fetches, feed_dict=feed_dict)
        beam_result.sort(key=lambda br: br[0], reverse=True)
        return beam_result[0][1]

    def __get_sequence(self):
        if self.__beam_size > 1:
            return self.__get_sequence_beam()
        sequence = []
        feed_dict = {self.__lstm.data: self.features, self.__lstm.word_input: np.ones(1),
                     self.__lstm.prev_c: np.zeros((1, 768)), self.__lstm.prev_h: np.zeros((1, 768)),
                     self.__lstm.flag: 1.0}
        fetches = [self.__lstm.prob, self.__lstm.next_c, self.__lstm.next_h]
        c = np.zeros((1, 768))
        h = np.zeros((1, 768))
        for i in range(self.__seq_length+2):
            if i == 1:
                feed_dict[self.__lstm.word_input] = self.__vocab_size*np.ones(1)
                feed_dict[self.__lstm.flag] = 0.0
            elif i > 1:
                feed_dict[self.__lstm.word_input] = np.argmax(prob, axis=1)*np.ones(1)
            feed_dict[self.__lstm.prev_c] = c
            feed_dict[self.__lstm.prev_h] = h
            prob, c, h = self.__lstm_sess.run(fetches=fetches, feed_dict=feed_dict)
            if i > 0:
                sequence.append(np.argmax(prob, axis=1)[0])

        return sequence

    def evaluate(self, img):
        img = util.img_preprocess(img)
        self.features, self.conv_features = self.__vgg_sess.run([self.__vgg.features, self.__vgg.conv_features],
                                                                feed_dict={self.__vgg.data: img})
        sequence = self.__get_sequence()
        self.__seq_postprocess(sequence)










