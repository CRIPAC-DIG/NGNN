import tensorflow as tf
import numpy as np
import json
from load_data_multimodal import load_num_category, load_graph, load_train_data, load_train_size, load_fitb_data, load_test_size, load_auc_data
#from load_data_2 import load_num_category, load_graph, load_train_data, load_valid_data, load_test_data
#from taobao_load_data import load_num_category, load_graph, load_train_data, load_valid_data, load_test_data
from model_multimodal_1 import GNN
from  datetime import *
import time
import pickle
from tensorflow.contrib.rnn import GRUCell
import os
##################load data###################
ftrain = open('train_no_dup_new_100.json', 'r')
train_outfit_list = json.load(ftrain)
ftest = open('test_no_dup_new_100.json', 'r')
test_outfit_list = json.load(ftest)

def cm_ggnn(batch_size, image_hidden_size, text_hidden_size, n_steps, learning_rate, G, num_category, opt, i, beta):

    hidden_stdv = np.sqrt(1. / (image_hidden_size))
    if i == 0:
        with tf.variable_scope("cm_ggnn", reuse=None):
            w_conf_image = tf.get_variable(name='gnn/w/conf_image', shape=[image_hidden_size, 1],
                                           initializer=tf.random_normal_initializer(hidden_stdv))
            w_score_image = tf.get_variable(name='gnn/w/score_image', shape=[image_hidden_size, 1],
                                            initializer=tf.random_normal_initializer(hidden_stdv))
            w_conf_text = tf.get_variable(name='gnn/w/conf_text', shape=[text_hidden_size, 1],
                                           initializer=tf.random_normal_initializer(hidden_stdv))
            w_score_text = tf.get_variable(name='gnn/w/score_text', shape=[text_hidden_size, 1],
                                            initializer=tf.random_normal_initializer(hidden_stdv))
            #w_atten = tf.get_variable(name='gnn/w/atten', shape=[num_category, num_category], initializer=tf.random_normal_initializer(num_stdv))
    else:
        with tf.variable_scope("cm_ggnn"):
            tf.get_variable_scope().reuse_variables()

    #################feed#######################
    image_pos = tf.placeholder(tf.float32, [batch_size, num_category, 2048])
    image_neg = tf.placeholder(tf.float32, [batch_size, num_category, 2048])
    text_pos = tf.placeholder(tf.float32, [batch_size, num_category, 2757])
    text_neg = tf.placeholder(tf.float32, [batch_size, num_category, 2757])
    graph_pos = tf.placeholder(tf.float32, [batch_size, num_category, num_category])
    graph_neg = tf.placeholder(tf.float32, [batch_size, num_category, num_category])

    ##################GGNN's output###################
    with tf.variable_scope("gnn_image", reuse=None):
        image_state_pos, image_ini = GNN('image', image_pos, batch_size, image_hidden_size, n_steps, num_category, graph_pos)  #output: [batch_size, num_category, 2048]
        tf.get_variable_scope().reuse_variables()
        image_state_neg, text_ini = GNN('image', image_neg, batch_size, image_hidden_size, n_steps, num_category, graph_neg)

    with tf.variable_scope("gnn_text", reuse=None):
        text_state_pos, test2 = GNN('text', text_pos, batch_size, text_hidden_size, n_steps, num_category, graph_pos)
        tf.get_variable_scope().reuse_variables()
        text_state_neg, test2 = GNN('text', text_neg, batch_size, text_hidden_size, n_steps, num_category, graph_neg)

    ##################predict positive###################
    for i in range(batch_size):

        image_conf_pos = tf.nn.sigmoid(tf.reshape(tf.matmul(image_state_pos[i], w_conf_image), [1, num_category]))
        image_score_pos = tf.reshape(tf.matmul(image_state_pos[i], w_score_image), [num_category, 1])
        image_score_pos = tf.maximum(0.01 * image_score_pos, image_score_pos)
        image_score_pos = tf.reshape(tf.matmul(image_conf_pos, image_score_pos), [1])
        text_conf_pos = tf.nn.sigmoid(tf.reshape(tf.matmul(text_state_pos[i], w_conf_text), [1, num_category]))
        text_score_pos = tf.reshape(tf.matmul(text_state_pos[i], w_score_text), [num_category, 1])
        text_score_pos = tf.maximum(0.01 * text_score_pos, text_score_pos)
        text_score_pos = tf.reshape(tf.matmul(text_conf_pos, text_score_pos), [1])
        score_pos = beta * image_score_pos + (1 - beta) * text_score_pos


        image_conf_neg = tf.nn.sigmoid(tf.reshape(tf.matmul(image_state_neg[i], w_conf_image), [1, num_category]))
        image_score_neg = tf.reshape(tf.matmul(image_state_neg[i], w_score_image), [num_category, 1])
        image_score_neg = tf.maximum(0.01 * image_score_neg, image_score_neg)
        image_score_neg = tf.reshape(tf.matmul(image_conf_neg, image_score_neg), [1])
        text_conf_neg = tf.nn.sigmoid(tf.reshape(tf.matmul(text_state_neg[i], w_conf_text), [1, num_category]))
        text_score_neg = tf.reshape(tf.matmul(text_state_neg[i], w_score_text), [num_category, 1])
        text_score_neg = tf.maximum(0.01 * text_score_neg, text_score_neg)
        text_score_neg = tf.reshape(tf.matmul(text_conf_neg, text_score_neg), [1])
        score_neg = beta * image_score_neg + (1 - beta) * text_score_neg

        if i == 0:
            s_pos = score_pos
            s_neg = score_neg
        else:
            s_pos = tf.concat([s_pos, score_pos], 0)
            s_neg = tf.concat([s_neg, score_neg], 0)

    s_pos = tf.reshape(s_pos, [batch_size, 1])
    s_neg = tf.reshape(s_neg, [batch_size, 1])

    s_pos_mean = tf.reduce_mean(s_pos)
    s_neg_mean = tf.reduce_mean(s_neg)

    ##################cost, optimizer###################
    cost_parameter = 0.
    num_parameter = 0.
    for variable in tf.trainable_variables():
        print (variable)
        cost_parameter += tf.contrib.layers.l2_regularizer(0.1)(variable)
        num_parameter += 1.
    cost_parameter /= num_parameter
    score = tf.nn.sigmoid(s_pos - s_neg)
    score_mean = tf.reduce_mean(score)
    cost_vt = tf.reduce_mean(tf.square(image_ini - text_ini))
    cost = -score_mean + 5 * cost_vt

    if opt == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    if opt == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)
    if opt == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    if opt == 'Adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)


    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    with tf.Session() as sess:

        # initialize the graph
        # 2017-03-02 if using tensorflow >= 0.12
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

        ######record######
        best_accurancy = 0.
        best_auc = 0.
        best_epoch = 0
        saver = tf.train.Saver()

        train_size, train_size_ = load_train_size()
        print ('train_size is %d' % train_size_)
        train_batch = int(train_size_ / batch_size)
        print ('train_batch is %d' % train_batch)

        for epoch in range(30):
            #########train##########
            test_interval = 2000
            if epoch > 10:
                test_interval = 1000

            no_count = 0
            c_all = 0.
            score_all = 0.
            vt_all = 0.
            dis_pos_all = 0.
            dis_neg_all = 0.
            for i in range(train_batch):
                train_image_pos, train_image_neg, train_text_pos, train_text_neg, \
                train_graph_pos, train_graph_neg, size_ = load_train_data(i, batch_size, train_outfit_list)
                if size_ >= batch_size:
                    image_pos_ = train_image_pos[0: batch_size]
                    image_neg_ = train_image_neg[0: batch_size]
                    text_pos_ = train_text_pos[0: batch_size]
                    text_neg_ = train_text_neg[0: batch_size]
                    train_graph_pos_ = train_graph_pos[0: batch_size]
                    train_graph_neg_ = train_graph_neg[0: batch_size]
                    # _, c, c_pred, dis_pos_, dis_neg_, conf_pos_, conf_neg_ = sess.run([optimizer, cost, cost_pred,
                    #                                                         dis_pos_mean, dis_neg_mean, conf_pos_mean, conf_neg_mean],
                    _, c, score, c_vt, dis_pos_, dis_neg_ = sess.run(
                            [optimizer, cost, score_mean, cost_vt,
                             s_pos_mean, s_neg_mean],
                                    feed_dict={image_pos: image_pos_,
                                               image_neg: image_neg_,
                                               text_pos: text_pos_,
                                               text_neg: text_neg_,
                                               graph_pos: train_graph_pos_,
                                               graph_neg: train_graph_neg_})
                    c_all += c
                    score_all += score
                    vt_all += c_vt
                    dis_pos_all += dis_pos_
                    dis_neg_all += dis_neg_


                    if i % test_interval == 0:
                        print ('now batch: %d, total batch: %d' % (i, train_batch))
                        print ('less than batch size: %d' % no_count)
                        c_average = c_all / (i + 1)
                        score_average = score_all / (i + 1)
                        vt_average = vt_all / (i + 1)
                        dis_pos_average = dis_pos_all / (i + 1)
                        dis_neg_average = dis_neg_all / (i + 1)

                        ############test############
                        test_size_fitb = load_test_size()
                        batches = int((test_size_fitb * 4) / batch_size)
                        right = 0.
                        for ii in range(batches):
                            test_fitb = load_fitb_data(ii, batch_size, test_outfit_list)
                            answer = sess.run([s_pos], feed_dict={image_pos: test_fitb[0],
                                                                                        text_pos: test_fitb[1],
                                                                                        graph_pos: test_fitb[2]})

                            answer = np.asarray(answer[0])

                            for j in range(batch_size / 4):
                                a = []
                                for k in range(j * 4, (j + 1) * 4):
                                    a.append(answer[k][0])
                                if np.argmax(a) == 0:
                                    right += 1.

                        print(answer)
                        accurancy = float(right / test_size_fitb)
                        if accurancy > best_accurancy:
                            best_accurancy = accurancy
                            best_epoch = epoch

                        ####### AUC #######
                        test_size_auc = load_test_size()
                        batches = int((test_size_auc * 2) / batch_size)
                        right = 0.
                        for ii in range(batches):
                            test_auc = load_auc_data(ii, batch_size, test_outfit_list)
                            answer = sess.run([s_pos], feed_dict={image_pos: test_auc[0],
                                                                    text_pos: test_auc[1],
                                                                    graph_pos: test_auc[2]})
                            answer = np.asarray(answer[0])

                            for j in range(batch_size / 2):
                                a = []
                                for k in range(j * 2, (j + 1) * 2):
                                    a.append(answer[k][0])
                                if np.argmax(a) == 0:
                                    right += 1.

                        print(answer)
                        auc = float(right / test_size_auc)

                        if auc > best_auc:
                            best_auc = auc
                            # saver.save(sess, "trained_model/cm_ggnn.ckpt")

                        print('now():' + str(datetime.now()))
                        print("Train Epoch:", '%d' % epoch, "Batch:", '%d' % i,
                              "total cost:", "{:.9f}".format(c_average), "pred score distance:",
                              "{:.9f}".format(score_average),
                              "vt cost:", "{:.9f}".format(vt_average), "postive score:",
                              "{:.9f}".format(dis_pos_average),
                              "negative score:", "{:.9f}".format(dis_neg_average),
                              "accurancy:", ".{:.9f}".format(accurancy), "auc:", ".{:.9f}".format(auc))
                        print("Epoch:", '%d' % epoch, "finished", "Best accurancy: %f" % best_accurancy,
                              "Best auc: %f" % best_auc,
                              "Best epoch: %d" % best_epoch)
                        print("batch_size: %d, image_hidden_size: %d, text_hidden_size: %d, n_steps: %d, learning_rate: %f" % (
                            batch_size, image_hidden_size, text_hidden_size, n_steps, learning_rate))

                else:
                    no_count += 1

            c_average = c_all / train_batch
            score_average = score_all / train_batch
            vt_average = vt_all / train_batch
            dis_pos_average = dis_pos_all / train_batch
            dis_neg_average = dis_neg_all / train_batch

            print("Train Epoch:", '%d' % epoch, "finished",
                  "total cost:", "{:.9f}".format(c_average), "pred score distance:", "{:.9f}".format(score_average),
                  "vt cost:", "{:.9f}".format(vt_average),
                  "postive score:", "{:.9f}".format(dis_pos_average), "negative score:",
                  "{:.9f}".format(dis_neg_average))
            print("Epoch:", '%d' % epoch, "finished", "Best accurancy: %f" % best_accurancy, "Best auc: %f" % best_auc,
                  "Best epoch: %d" % best_epoch)
            print("batch_size: %d, image_hidden_size: %d, image_hidden_size: %d, n_steps: %d, learning_rate: %f" % (
                batch_size, image_hidden_size, text_hidden_size, n_steps, learning_rate))

            ############test############
            batches = int((test_size_fitb * 4) / batch_size)
            right = 0.
            for i in range(batches):
                test_fitb = load_fitb_data(i, batch_size, test_outfit_list)
                answer = sess.run([s_pos], feed_dict={image_pos: test_fitb[0],
                                            text_pos: test_fitb[1],
                                            graph_pos: test_fitb[2]})

                answer = np.asarray(answer[0])

                for j in range(batch_size / 4):
                    a = []
                    for k in range(j * 4, (j + 1) * 4):
                        a.append(answer[k][0])
                    if np.argmax(a) == 0:
                        right += 1.
            print(answer)
            accurancy = float(right / test_size_fitb)

            ##### AUC #####
            batches = int((test_size_auc * 2) / batch_size)
            right = 0.
            for i in range(batches):
                test_auc = load_auc_data(i, batch_size, test_outfit_list)
                answer = sess.run([s_pos], feed_dict={image_pos: test_auc[0],
                                                       text_pos: test_auc[1],
                                                       graph_pos: test_auc[2]})
                answer = np.asarray(answer[0])

                for j in range(batch_size / 2):
                    a = []
                    for k in range(j * 2, (j + 1) * 2):
                        a.append(answer[k][0])
                    if np.argmax(a) == 0:
                        right += 1.

            print(answer)
            auc = float(right / test_size_auc)

            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch

            if accurancy > best_accurancy:
                best_accurancy = accurancy
                best_epoch = epoch
                saver.save(sess, "multi_modal_1/cm_ggnn.ckpt")

            print("Test Epoch:", '%d' % epoch, "accuracy:", "{:.9f}".format(accurancy), "auc:", "{:.9f}".format(auc))

            print('now():' + str(datetime.now()))
            print("batch_size: %d, image_hidden_size: %d, text_hidden_size: %d, n_steps: %d, learning_rate: %f" % (
                batch_size, image_hidden_size, text_hidden_size, n_steps, learning_rate))
            print("Epoch:", '%d' % epoch, "finished", "Best accurancy: %f" % best_accurancy, "Best auc: %f" % best_auc,
                  "Best epoch: %d" % best_epoch)

    return best_accurancy


def look_enable_node(graph):
    if_enable = np.sum(graph, axis=1)
    index_list = []
    for index, value in enumerate(if_enable):
        if value > 0:
            index_list.append(index)
    return index_list


if __name__ == '__main__':

    num_category = load_num_category()
    G = load_graph()
    best_accurancy = 0.
    i = 0
    batch_size = 16
    image_hidden_size = 12
    text_hidden_size = 12
    n_steps = 3
    learning_rate = 0.001
    opt = RMSProp"
    beta = 0.2
    accurancy = cm_ggnn(batch_size, image_hidden_size, text_hidden_size, n_steps, learning_rate, G, num_category, opt, i, beta)
#     for image_hidden_size in [12]: #### n*8
#         for text_hidden_size in [12, 16, 64]:
#             for n_steps in [3]:
#                 for learning_rate in [0.001]:
#                     for opt in ['RMSProp', 'Adam']:
#                         for beta in [0.2, 0.5, 0.7]:
#                             accurancy = cm_ggnn(batch_size, image_hidden_size, text_hidden_size, n_steps, learning_rate, G, num_category, opt, i, beta)
#                             if accurancy > best_accurancy:
#                                 best_accurancy = accurancy
#                                 best_parameter = [batch_size, image_hidden_size, text_hidden_size, n_steps, learning_rate]
#                                 print("best parameter is batch_size, image_hidden_size, text_hidden_size, n_steps, learning_rate, optimizer:%d, %d ,%d , %d, %f, %s" % (batch_size,
#                                                                                                                         image_hidden_size, text_hidden_size, n_steps, learning_rate, opt))
#                                 i += 1


