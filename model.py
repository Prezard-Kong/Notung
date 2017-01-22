from __future__ import print_function
from __future__ import unicode_literals
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
import util
import os
import numpy as np


class Vgg16(object):

    def __init__(self, batch_size=1):
        self.__batch_size = batch_size
        self.varlist = []
        self.__build()

    def __conv_relu(self, input, shape, index):
        weight = tf.Variable(tf.truncated_normal(shape=shape), name='weight{}'.format(index))
        bias = tf.Variable(tf.zeros((shape[-1])), name='bias{}'.format(index))
        self.varlist.append(weight)
        self.varlist.append(bias)
        conv = tf.nn.conv2d(input, weight, [1, 1, 1, 1], padding='SAME', name='conv{}'.format(index))
        return tf.nn.relu(conv + bias, name='relu{}'.format(index))

    def __build(self):
        pool_size = (1, 2, 2, 1)
        graph = tf.Graph()
        self.graph = graph
        with graph.as_default():
            data = tf.placeholder(tf.float32, shape=(self.__batch_size, 224, 224, 3), name='data')
            self.data = data
            relu1 = self.__conv_relu(data, (3, 3, 3, 64), 1)
            relu2 = self.__conv_relu(relu1, (3, 3, 64, 64), 2)
            pool2 = tf.nn.max_pool(relu2, pool_size, pool_size, padding='VALID', name='pool2')
            relu3 = self.__conv_relu(pool2, (3, 3, 64, 128), 3)
            relu4 = self.__conv_relu(relu3, (3, 3, 128, 128), 4)
            pool4 = tf.nn.max_pool(relu4, pool_size, pool_size, padding='VALID', name='pool4')
            relu5 = self.__conv_relu(pool4, (3, 3, 128, 256), 5)
            relu6 = self.__conv_relu(relu5, (3, 3, 256, 256), 6)
            relu7 = self.__conv_relu(relu6, (3, 3, 256, 256), 7)
            pool7 = tf.nn.max_pool(relu7, pool_size, pool_size, padding='VALID', name='pool7')
            relu8 = self.__conv_relu(pool7, (3, 3, 256, 512), 8)
            relu9 = self.__conv_relu(relu8, (3, 3, 512, 512), 9)
            relu10 = self.__conv_relu(relu9, (3, 3, 512, 512), 10)
            pool10 = tf.nn.max_pool(relu10, pool_size, pool_size, padding='VALID', name='pool10')
            relu11 = self.__conv_relu(pool10, (3, 3, 512, 512), 11)
            relu12 = self.__conv_relu(relu11, (3, 3, 512, 512), 12)
            relu13 = self.__conv_relu(relu12, (3, 3, 512, 512), 13)
            pool13 = tf.nn.max_pool(relu13, pool_size, pool_size, padding='VALID', name='pool13')
            trans = tf.transpose(pool13, perm=[0, 3, 1, 2])  # transpose to fit the tensor's shape in Torch
            weight14 = tf.Variable(tf.truncated_normal((25088, 4096)), name='weight14')
            bias14 = tf.Variable(tf.zeros(4096), name='bias14')
            self.varlist.append(weight14)
            self.varlist.append(bias14)
            flat = tf.reshape(trans, shape=(self.__batch_size, -1), name='flat')
            self.conv_features = flat
            fc14 = tf.nn.xw_plus_b(flat, weight14, bias14, name='fc14')
            relu14 = tf.nn.relu(fc14, name='relu14')
            weight15 = tf.Variable(tf.truncated_normal((4096, 4096)), name='weight15')
            bias15 = tf.Variable(tf.zeros(4096), name='bias15')
            self.varlist.append(weight15)
            self.varlist.append(bias15)
            fc15 = tf.nn.xw_plus_b(relu14, weight15, bias15, name='fc15')
            relu15 = tf.nn.relu(fc15, name='relu15')
            weight16 = tf.Variable(tf.truncated_normal((4096, 768)), name='weight16')
            bias16 = tf.Variable(tf.zeros(768), name='bias16')
            self.varlist.append(weight16)
            self.varlist.append(bias16)
            fc16 = tf.nn.xw_plus_b(relu15, weight16, bias16, name='fc16')
            relu16 = tf.nn.relu(fc16, name='relu16')
            self.features = relu16


class LSTM(object):

    def __init__(self, input_size=768, output_size=9568, rnn_size=768, num_layers=1, batch_size=1):
        self.__input_size = input_size
        self.__output_size = output_size
        self.__rnn_size = rnn_size
        self.__num_layers = num_layers
        self.__batch_size = batch_size
        self.varlist = []
        self.__build()

    def __i2h(self, index):
        weight = tf.Variable(tf.truncated_normal((self.__input_size, self.__rnn_size*4)),
                             name='i2h_weight{}'.format(index))
        bias = tf.Variable(tf.zeros(self.__rnn_size*4), name='i2h_bias{}'.format(index))
        self.varlist.append(weight)
        self.varlist.append(bias)
        return weight, bias

    def __h2h(self, index):
        weight = tf.Variable(tf.truncated_normal((self.__rnn_size, self.__rnn_size * 4)),
                             name='h2h_weight{}'.format(index))
        bias = tf.Variable(tf.zeros(self.__rnn_size*4), name='h2h_bias{}'.format(index))
        self.varlist.append(weight)
        self.varlist.append(bias)
        return weight, bias

    def __build(self):
        graph = tf.Graph()
        self.graph = graph
        with graph.as_default():
            self.data = tf.placeholder(dtype=tf.float32, shape=(self.__batch_size, self.__input_size), name='data')
            self.word_input = tf.placeholder(dtype=tf.int32, shape=self.__batch_size, name='input')
            i2h = []
            h2h = []
            for i in range(self.__num_layers):
                i2h.append(self.__i2h(i+1))
                h2h.append(self.__h2h(i+1))
            output = tf.placeholder(dtype=tf.float32, shape=(self.__batch_size, self.__rnn_size), name='prev_h')
            state = tf.placeholder(dtype=tf.float32, shape=(self.__batch_size, self.__rnn_size), name='prev_c')
            self.prev_h = output
            self.prev_c = state

            # classifier
            weight = tf.Variable(tf.truncated_normal((self.__rnn_size, self.__output_size)), name='proj_weight')
            bias = tf.Variable(tf.zeros(self.__output_size), name='proj_bias')
            self.varlist.append(weight)
            self.varlist.append(bias)
            # word embedding
            embedding = tf.Variable(tf.random_normal((self.__output_size, self.__rnn_size)), name='embedding')
            self.varlist.append(embedding)

            def lstm_cell(input, output, state):
                for i in range(self.__num_layers):
                    input_ = input if i == 0 else output
                    gate = tf.matmul(input_, i2h[i][0])+i2h[i][1]+tf.matmul(output, h2h[i][0])+h2h[i][1]
                    in_gate = tf.sigmoid(gate[:, :self.__rnn_size])
                    forget_gate = tf.sigmoid(gate[:, self.__rnn_size:2*self.__rnn_size])
                    out_gate = tf.sigmoid(gate[:, 2*self.__rnn_size:3*self.__rnn_size])
                    in_transform = tf.tanh(gate[:, 3*self.__rnn_size:])
                    state = tf.mul(forget_gate, state)+tf.mul(in_gate, in_transform)
                    output = tf.mul(out_gate, tf.tanh(state))
                return output, state

            self.flag = tf.placeholder(dtype=tf.float32, name='flag')
            inp = tf.cond(self.flag > tf.constant(0.5), lambda: self.data,
                          lambda: tf.nn.embedding_lookup(embedding, self.word_input))
            self.next_h, self.next_c = lstm_cell(inp, output, state)
            self.prob = tf.nn.log_softmax(tf.nn.xw_plus_b(self.next_h, weight, bias))


def content_loss(tensor1, tensor2):
    return tf.nn.l2_loss(tf.sub(tensor1, tensor2))


def style_loss(tensor1, tensor2):
    shape = tf.shape(tensor1, out_type=tf.float32)
    h = shape[1]
    w = shape[2]
    d = shape[3]
    a = tf.transpose(tf.reshape(tensor1, (-1, tf.to_int32(d))))
    b = tf.transpose(tf.reshape(tensor2, (-1, tf.to_int32(d))))
    gram1 = tf.matmul(a, a, transpose_b=True)
    gram2 = tf.matmul(b, b, transpose_b=True)
    return (tf.nn.l2_loss(tf.sub(gram1, gram2))/d**2/(h*w)**2)*0.1


def tv_loss(img):
    tv_x = tf.nn.l2_loss(img[:, :, 1:, :]-img[:, :, :-1, :])
    tv_y = tf.nn.l2_loss(img[:, 1:, :, :]-img[:, :-1, :, :])
    return (tv_x+tv_y)*2


class NeuralStyle(object):

    def __init__(self, img_width=512, img_height=512, content_weight=5.0, style_weight=1e4, tv_weight=1e-3,
                 SGD=False, max_iters=1000, learning_rate=1.0):
        self.__img_width = img_width
        self.__img_height = img_height
        self.__content_weight = content_weight
        self.__style_weight = style_weight
        self.__tv_weight = tv_weight
        self.__SGD = SGD
        self.__max_iters = max_iters
        self.__learning_rate = learning_rate
        self.__varlist = []
        self.__style_p = []
        self.__style = []
        self.__build()

    def __conv_relu(self, input, shape, index):
        weight = tf.Variable(tf.truncated_normal(shape=shape), name='weight{}'.format(index), trainable=False)
        bias = tf.Variable(tf.zeros((shape[-1])), name='bias{}'.format(index), trainable=False)
        self.__varlist.append(weight)
        self.__varlist.append(bias)
        conv = tf.nn.conv2d(input, weight, [1, 1, 1, 1], padding='SAME', name='conv{}'.format(index))
        return tf.nn.relu(conv + bias, name='relu{}'.format(index))

    def __build(self):
        pool_size = (1, 2, 2, 1)
        graph = tf.Graph()
        self.graph = graph
        with graph.as_default():
            tv_loss_val = 0
            content_loss_val = 0
            style_loss_val = 0
            # flag to control the flow, False to get the result, True to get image features
            self.flag = tf.placeholder(tf.bool, name='flag')
            # content features
            self.content_p = tf.placeholder(tf.float32, name='content')  # relu4_2
            # style features
            self.__style_p.append(tf.placeholder(tf.float32, name='style1'))  # relu1_1
            self.__style_p.append(tf.placeholder(tf.float32, name='style2'))  # relu2_1
            self.__style_p.append(tf.placeholder(tf.float32, name='style3'))  # relu3_1
            self.__style_p.append(tf.placeholder(tf.float32, name='style4'))  # relu4_1
            self.__style_p.append(tf.placeholder(tf.float32, name='style5'))  # relu5_1

            self.__result = tf.Variable(tf.random_normal(shape=(1, self.__img_height, self.__img_width, 3)),
                                        name='result')
            self.__img = tf.placeholder(tf.float32, shape=(1, self.__img_height, self.__img_width, 3), name='img')
            tv_loss_val += tv_loss(self.__img)
            data = tf.cond(tf.equal(self.flag, False), lambda: self.__result, lambda: self.__img)
            relu1 = self.__conv_relu(data, (3, 3, 3, 64), 1)
            self.__style.append(relu1)
            style_loss_val += style_loss(self.__style[0], self.__style_p[0])
            relu2 = self.__conv_relu(relu1, (3, 3, 64, 64), 2)
            pool2 = tf.nn.avg_pool(relu2, pool_size, pool_size, padding='VALID', name='pool2')
            relu3 = self.__conv_relu(pool2, (3, 3, 64, 128), 3)
            self.__style.append(relu3)
            style_loss_val += style_loss(self.__style[1], self.__style_p[1])
            relu4 = self.__conv_relu(relu3, (3, 3, 128, 128), 4)
            pool4 = tf.nn.avg_pool(relu4, pool_size, pool_size, padding='VALID', name='pool4')
            relu5 = self.__conv_relu(pool4, (3, 3, 128, 256), 5)
            self.__style.append(relu4)
            style_loss_val += style_loss(self.__style[2], self.__style_p[2])
            relu6 = self.__conv_relu(relu5, (3, 3, 256, 256), 6)
            relu7 = self.__conv_relu(relu6, (3, 3, 256, 256), 7)
            pool7 = tf.nn.avg_pool(relu7, pool_size, pool_size, padding='VALID', name='pool7')
            relu8 = self.__conv_relu(pool7, (3, 3, 256, 512), 8)
            self.__style.append(relu8)
            style_loss_val += style_loss(self.__style[3], self.__style_p[3])
            relu9 = self.__conv_relu(relu8, (3, 3, 512, 512), 9)
            self.__content = relu9
            content_loss_val += content_loss(self.__content, self.content_p)
            relu10 = self.__conv_relu(relu9, (3, 3, 512, 512), 10)
            pool10 = tf.nn.avg_pool(relu10, pool_size, pool_size, padding='VALID', name='pool10')
            relu11 = self.__conv_relu(pool10, (3, 3, 512, 512), 11)
            self.__style.append(relu11)
            style_loss_val += style_loss(self.__style[4], self.__style_p[4])
            relu12 = self.__conv_relu(relu11, (3, 3, 512, 512), 12)
            relu13 = self.__conv_relu(relu12, (3, 3, 512, 512), 13)
            pool13 = tf.nn.avg_pool(relu13, pool_size, pool_size, padding='VALID', name='pool13')

            self.__loss = self.__content_weight*content_loss_val \
                + self.__style_weight*style_loss_val + self.__tv_weight*tv_loss_val

            self.__opt = tf.train.AdamOptimizer(learning_rate=self.__learning_rate).minimize(self.__loss)

    def evaluate(self, content_path, style_path, save_path='./data/img.jpg'):
        content = util.img_preprocess(util.load_img(content_path), self.__img_width, self.__img_height, crop=False)
        style = util.img_preprocess(util.load_img(style_path), self.__img_width, self.__img_height, crop=False)
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_list=self.__varlist)
            saver.restore(sess, './data/cnn/cnn.ckpt')
            feed_dict = {self.flag: True, self.__img: content}
            content_features = sess.run(self.__content, feed_dict=feed_dict)
            feed_dict[self.__img] = style
            style_features = sess.run(self.__style, feed_dict=feed_dict)
            for i in range(5):
                feed_dict[self.__style_p[i]] = style_features[i]
            feed_dict[self.content_p] = content_features
            feed_dict[self.flag] = False
            if self.__SGD:
                for i in range(self.__max_iters):
                    _, result = sess.run([self.__opt, self.__result], feed_dict=feed_dict)
            else:
                opt = ScipyOptimizerInterface(self.__loss, method='L-BFGS-B', options={'maxiter': self.__max_iters})
                opt.minimize(session=sess, feed_dict=feed_dict, fetches=[self.__result])
                result = sess.run(self.__result)
            img = util.img_deprocess(result)
            self.transformed_img = img
            img.save(save_path)


class FastNeuralStyle(object):

    def __init__(self, batch_size=1, img_width=512, img_height=512, content_weight=5.0,
                 style_weight=1e4, tv_weight=1e-5, learning_rate=1e-3, decay_steps=20000):
        self.__batch_size = batch_size
        self.__img_width = img_width
        self.__img_height = img_height
        self.__content_weight = content_weight
        self.__style_weight = style_weight
        self.__tv_weight = tv_weight
        self.__learning_rate = learning_rate
        self.__decay_steps = decay_steps
        self.__varlist_loss_net = []
        self.__varlist_trans_net = []
        self.__style = []
        self.__style_p = []
        self.__build()

    def __conv_relu_vgg(self, input, shape, index):
        weight = tf.Variable(tf.truncated_normal(shape=shape), name='weight{}'.format(index), trainable=False)
        bias = tf.Variable(tf.zeros((shape[-1])), name='bias{}'.format(index), trainable=False)
        self.__varlist_loss_net.append(weight)
        self.__varlist_loss_net.append(bias)
        conv = tf.nn.conv2d(input, weight, [1, 1, 1, 1], padding='SAME', name='conv{}'.format(index))
        return tf.nn.relu(conv + bias, name='relu{}'.format(index))

    def __conv_relu(self, input, shape, stride, trainable=True, relu=True):
        weight = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name='weight', trainable=trainable)
        self.__varlist_trans_net.append(weight)
        conv = tf.nn.conv2d(input, weight, [1, stride, stride, 1], padding='SAME', name='conv')
        norm = self.__instance_norm(conv)
        if relu:
            return tf.nn.relu(norm, name='relu')
        else:
            return norm

    def __deconv_relu(self, input, shape, stride, relu=True):
        weight = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name='weight')
        self.__varlist_trans_net.append(weight)
        b, h, w, d = input.get_shape().as_list()
        out_shape = tf.pack([b, h*stride, w*stride, shape[2]])
        deconv = tf.nn.conv2d_transpose(input, weight, out_shape, [1, stride, stride, 1], padding='SAME', name='deconv')
        norm = self.__instance_norm(deconv)
        if relu:
            return tf.nn.relu(norm, name='relu')
        else:
            return norm

    def __residual_block(self, input, shape):
        conv = self.__conv_relu(input, shape, 1)
        return input+self.__conv_relu(conv, shape, 1, relu=False)

    def __instance_norm(self, input):
        input_shape = input.get_shape().as_list()
        mean, variance = tf.nn.moments(input, (1, 2), keep_dims=True)
        norm = (input-mean)/tf.sqrt(variance+1e-6)
        scale = tf.Variable(tf.ones((input_shape[3])), name='scale')
        shift = tf.Variable(tf.zeros((input_shape[3])), name='shift')
        self.__varlist_trans_net.append(scale)
        self.__varlist_trans_net.append(shift)
        return scale*norm+shift

    def __build(self):
        pool_size = (1, 2, 2, 1)
        graph = tf.Graph()
        self.graph = graph
        with graph.as_default():
            self.__img = tf.placeholder(dtype=tf.float32, shape=(self.__batch_size, self.__img_height,
                                                                 self.__img_width, 3), name='content')
            conv1 = self.__conv_relu(self.__img, (9, 9, 3, 32), 1)
            conv2 = self.__conv_relu(conv1, (3, 3, 32, 64), 2)
            conv3 = self.__conv_relu(conv2, (3, 3, 64, 128), 2)
            res1 = self.__residual_block(conv3, (3, 3, 128, 128))
            res2 = self.__residual_block(res1, (3, 3, 128, 128))
            res3 = self.__residual_block(res2, (3, 3, 128, 128))
            res4 = self.__residual_block(res3, (3, 3, 128, 128))
            res5 = self.__residual_block(res4, (3, 3, 128, 128))
            deconv1 = self.__deconv_relu(res5, (3, 3, 64, 128), 2)
            deconv2 = self.__deconv_relu(deconv1, (3, 3, 32, 64), 2)
            conv_f = self.__conv_relu(deconv2, (3, 3, 32, 3), 1, relu=False)
            self.__transformed_img = tf.tanh(conv_f)*150

            tv_loss_val = 0
            content_loss_val = 0
            style_loss_val = 0
            # flag to control the flow, True to get the result, False to get image features
            self.__flag = tf.placeholder(tf.bool, name='flag')
            # content features
            self.content_p = tf.placeholder(tf.float32, name='content')  # relu4_2
            # style features
            self.__style_p.append(tf.placeholder(tf.float32, name='style1'))  # relu1_1
            self.__style_p.append(tf.placeholder(tf.float32, name='style2'))  # relu2_1
            self.__style_p.append(tf.placeholder(tf.float32, name='style3'))  # relu3_1
            self.__style_p.append(tf.placeholder(tf.float32, name='style4'))  # relu4_1
            self.__style_p.append(tf.placeholder(tf.float32, name='style5'))  # relu5_1

            tv_loss_val += tv_loss(self.__transformed_img)
            data = tf.cond(tf.equal(self.__flag, False), lambda: self.__transformed_img, lambda: self.__img)
            relu1 = self.__conv_relu_vgg(data, (3, 3, 3, 64), 1)
            self.__style.append(relu1)
            style_loss_val += style_loss(self.__style[0], self.__style_p[0])
            relu2 = self.__conv_relu_vgg(relu1, (3, 3, 64, 64), 2)
            pool2 = tf.nn.avg_pool(relu2, pool_size, pool_size, padding='VALID', name='pool2')
            relu3 = self.__conv_relu(pool2, (3, 3, 64, 128), 3)
            self.__style.append(relu3)
            style_loss_val += style_loss(self.__style[1], self.__style_p[1])
            relu4 = self.__conv_relu_vgg(relu3, (3, 3, 128, 128), 4)
            pool4 = tf.nn.avg_pool(relu4, pool_size, pool_size, padding='VALID', name='pool4')
            relu5 = self.__conv_relu_vgg(pool4, (3, 3, 128, 256), 5)
            self.__style.append(relu4)
            style_loss_val += style_loss(self.__style[2], self.__style_p[2])
            relu6 = self.__conv_relu_vgg(relu5, (3, 3, 256, 256), 6)
            relu7 = self.__conv_relu_vgg(relu6, (3, 3, 256, 256), 7)
            pool7 = tf.nn.avg_pool(relu7, pool_size, pool_size, padding='VALID', name='pool7')
            relu8 = self.__conv_relu_vgg(pool7, (3, 3, 256, 512), 8)
            self.__style.append(relu8)
            style_loss_val += style_loss(self.__style[3], self.__style_p[3])
            relu9 = self.__conv_relu_vgg(relu8, (3, 3, 512, 512), 9)
            self.__content = relu9
            content_loss_val += content_loss(self.__content, self.content_p)
            relu10 = self.__conv_relu_vgg(relu9, (3, 3, 512, 512), 10)
            pool10 = tf.nn.avg_pool(relu10, pool_size, pool_size, padding='VALID', name='pool10')
            relu11 = self.__conv_relu_vgg(pool10, (3, 3, 512, 512), 11)
            self.__style.append(relu11)
            style_loss_val += style_loss(self.__style[4], self.__style_p[4])
            relu12 = self.__conv_relu_vgg(relu11, (3, 3, 512, 512), 12)
            relu13 = self.__conv_relu_vgg(relu12, (3, 3, 512, 512), 13)
            pool13 = tf.nn.avg_pool(relu13, pool_size, pool_size, padding='VALID', name='pool13')

            self.__loss = self.__content_weight * content_loss_val \
                          + self.__style_weight * style_loss_val + self.__tv_weight * tv_loss_val
            self.__log = tf.summary.scalar('loss', self.__loss)

            self.__global_step = tf.placeholder(tf.int32)
            learning_rate = tf.train.exponential_decay(learning_rate=self.__learning_rate,
                                                       global_step=self.__global_step,
                                                       decay_steps=self.__decay_steps,
                                                       decay_rate=0.1)

            self.__opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.__loss)

    def train(self, content_path, style_path, log=False, step=100):
        paths = [os.path.join(content_path, img) for img in os.listdir(content_path)]
        np.random.shuffle(paths)
        num = len(paths)/self.__batch_size-1
        style = util.img_preprocess(util.load_img(style_path), self.__img_width, self.__img_height, crop=False)
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver1 = tf.train.Saver(var_list=self.__varlist_loss_net)
            saver1.restore(sess, './data/cnn/cnn.ckpt')
            feed_dict = {self.__flag: True, self.__img: np.tile(style, (self.__batch_size, 1, 1, 1))}
            style_features = sess.run(self.__style, feed_dict=feed_dict)
            if log:
                writer = tf.summary.FileWriter('./data/log', graph=self.graph)
            for i in range(num*2):  # 2 epochs
                batch = util.get_batch(paths, (i % num)*self.__batch_size, batch_size=self.__batch_size,
                                       width=self.__img_width, height=self.__img_height)
                feed_dict[self.__img] = batch
                feed_dict[self.__flag] = True
                content_features = sess.run(self.__content, feed_dict=feed_dict)
                feed_dict[self.__flag] = False
                for j in range(5):
                    feed_dict[self.__style_p[j]] = style_features[j]
                feed_dict[self.content_p] = content_features
                feed_dict[self.__global_step] = i
                loss_log, loss, _ = sess.run([self.__log, self.__loss, self.__opt], feed_dict=feed_dict)
                if log and i%step == 0:
                    writer.add_summary(loss_log, i)
                    print('loss at iterations {}: {}'.format(i, loss))
            saver2 = tf.train.Saver(var_list=self.__varlist_trans_net)
            if not os.path.exists('./data/neural_style'):
                os.mkdir('./data/neural_style')
            saver2.save(sess, './data/neural_style/neural_style.ckpt')

    def evaluate(self, img_path, save_path='./data/img.jpg'):
        img = util.img_preprocess(util.load_img(img_path), self.__img_width, self.__img_height, crop=False)
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver(var_list=self.__varlist_trans_net)
            saver.restore(sess, './data/neural_style/neural_style.ckpt')
            feed_dict = {self.__img: img}
            transformed_img = sess.run(self.__transformed_img, feed_dict=feed_dict)
            self.transformed_img = util.img_deprocess(transformed_img)
            self.transformed_img.save(save_path)















