import tensorflow as tf
import numpy as np
from util.load_data_score_graph import load_num_category, load_graph, load_train_data, load_train_size
from model.model_score_in_out import GNN
from  datetime import *
import pickle
import os
##################load data###################


read_file_fill = open('fill_in_blank_1000_from_test_score.pkl', 'rb')
test_image, test_graph, test_size = pickle.load(read_file_fill)


def cm_ggnn(batch_size, hidden_size, n_steps, learning_rate, G, num_category, opt, i, beta):

    hidden_stdv = np.sqrt(1. / (hidden_size))
    if i == 0:
        with tf.variable_scope("cm_ggnn", reuse=None):
            # w_conf1 = tf.Variable(tf.random_normal([2048+hidden_size, hidden_size]), name='gnn/w/conf_1')
            w_conf2 = tf.get_variable(name='gnn/w/conf_2', shape=[hidden_size, 1], initializer=tf.random_normal_initializer(hidden_stdv))
            # w_score1 = tf.Variable(tf.random_normal([2048 + hidden_size, hidden_size]), name='gnn/w/score_1')
            w_score2 = tf.get_variable(name='gnn/w/score_2', shape=[hidden_size, 1], initializer=tf.random_normal_initializer(hidden_stdv))
    else:
        tf.get_variable_scope().reuse_variables()

    #################feed#######################
    image_pos = tf.placeholder(tf.float32, [batch_size, num_category, 2048])
    image_neg = tf.placeholder(tf.float32, [batch_size, num_category, 2048])
    graph_pos = tf.placeholder(tf.float32, [batch_size, num_category, num_category])
    graph_neg = tf.placeholder(tf.float32, [batch_size, num_category, num_category])

    ##################GGNN's output###################
    with tf.variable_scope("gnn", reuse=None):
        state_pos, test1 = GNN(image_pos, batch_size, hidden_size, n_steps, num_category, graph_pos)  #output: [batch_size, num_category, 2048]
        tf.get_variable_scope().reuse_variables()
        state_neg, test2 = GNN(image_neg, batch_size, hidden_size, n_steps, num_category, graph_neg)

    ##################predict positive###################
    for j in range(num_category):
        # state_image_pos = tf.concat([tf.reshape(state_pos[:, j, :], [-1, hidden_size]),
        #                              tf.reshape(image_pos[:, j, :], [-1, 2048])], 1)
        # conf_pos = tf.matmul(state_image_pos, w_conf1)
        # conf_pos = tf.nn.tanh(conf_pos)
        # conf_pos = tf.reshape(tf.matmul(conf_pos, w_conf2), [-1])
        # conf_pos = tf.nn.sigmoid(conf_pos)
        conf_pos = tf.matmul(tf.reshape(state_pos[:, j, :], [-1, hidden_size]), w_conf2)
        conf_pos = tf.nn.sigmoid(conf_pos)

        # score_pos = tf.matmul(state_image_pos, w_score1)
        # score_pos = tf.nn.tanh(score_pos)
        # score_pos = tf.reshape(tf.matmul(score_pos, w_score2), [-1])
        # score_pos = tf.nn.tanh(score_pos)
        score_pos = tf.matmul(tf.reshape(state_pos[:, j, :], [-1, hidden_size]), w_score2)
        # score_pos = tf.nn.relu(score_pos)
        score_pos = tf.maximum(0.01 * score_pos, score_pos)

        # state_image_neg = tf.concat([tf.reshape(state_neg[:, j, :], [-1, hidden_size]),
        #                              tf.reshape(image_neg[:, j, :], [-1, 2048])], 1)
        # conf_neg = tf.matmul(state_image_neg, w_conf1)
        # conf_neg = tf.nn.tanh(conf_neg)
        # conf_neg = tf.reshape(tf.matmul(conf_neg, w_conf2), [-1])
        # conf_neg = tf.nn.sigmoid(conf_neg)
        conf_neg = tf.matmul(tf.reshape(state_neg[:, j, :], [-1, hidden_size]), w_conf2)
        conf_neg = tf.nn.sigmoid(conf_neg)

        # score_neg = tf.matmul(state_image_neg, w_score1)
        # score_neg = tf.nn.tanh(score_neg)
        # score_neg = tf.reshape(tf.matmul(score_neg, w_score2), [-1])
        # score_neg = tf.nn.tanh(score_neg)
        score_neg = tf.matmul(tf.reshape(state_neg[:, j, :], [-1, hidden_size]), w_score2)
        # score_neg = tf.nn.relu(score_neg)
        score_neg = tf.maximum(0.01 * score_neg, score_neg)


        if j == 0:
            s_pos = score_pos * conf_pos
            s_neg = score_neg * conf_neg
        else:
            s_pos += score_pos * conf_pos
            s_neg += score_neg * conf_neg

    s_pos = tf.reshape(s_pos, [-1, 1])
    s_neg = tf.reshape(s_neg, [-1, 1])

    s_pos = tf.reshape(s_pos, [batch_size, 1])
    s_neg = tf.reshape(s_neg, [batch_size, 1])
    s_pos_mean = tf.reduce_mean(s_pos)
    s_neg_mean = tf.reduce_mean(s_neg)
    ##################cost, optimizer###################
    cost_parameter = 0.
    num_parameter = 0.
    for variable in tf.trainable_variables():
        print (variable)
        cost_parameter += tf.contrib.layers.l2_regularizer(beta)(variable)
        num_parameter += 1.
    cost_parameter /= num_parameter
    score = tf.nn.sigmoid(s_pos - s_neg)
    score_mean = tf.reduce_mean(score)
    cost = -score_mean

    if opt == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    if opt == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)
    if opt == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    if opt == 'Adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)


    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
        best_epoch = 0
        saver = tf.train.Saver()

        train_size, train_size_ = load_train_size()
        print ('train_size is %d' % train_size_)
        train_batch = int(train_size_ / batch_size)
        print ('train_batch is %d' % train_batch)

        for epoch in range(20):
            #########train##########
            no_count = 0
            c_all = 0.
            score_all = 0.
            para_all = 0.
            dis_pos_all = 0.
            dis_neg_all = 0.
            for i in range(train_batch):
                train_image_pos, train_image_neg, train_graph_pos, train_graph_neg, size_ = load_train_data(i, batch_size)
                if size_ >= batch_size:
                    image_pos_ = train_image_pos[0: batch_size]
                    image_neg_ = train_image_neg[0: batch_size]
                    train_graph_pos_ = train_graph_pos[0: batch_size]
                    train_graph_neg_ = train_graph_neg[0: batch_size]
                    # _, c, c_pred, dis_pos_, dis_neg_, conf_pos_, conf_neg_ = sess.run([optimizer, cost, cost_pred,
                    #                                                         dis_pos_mean, dis_neg_mean, conf_pos_mean, conf_neg_mean],
                    _, c, score, c_parameter, dis_pos_, dis_neg_ = sess.run(
                            [optimizer, cost, score_mean, cost_parameter,
                             s_pos_mean, s_neg_mean],
                                    feed_dict={image_pos: image_pos_,
                                               image_neg: image_neg_,
                                               graph_pos: train_graph_pos_,
                                               graph_neg: train_graph_neg_})
                    c_all += c
                    score_all += score
                    para_all += c_parameter
                    dis_pos_all += dis_pos_
                    dis_neg_all += dis_neg_

                    if i % 2000 == 0:
                        print ('now batch: %d, total batch: %d' % (i, train_batch))
                        print ('less than batch size: %d' % no_count)
                        c_average = c_all / (i + 1)
                        score_average = score_all / (i + 1)
                        para_average = para_all / (i + 1)
                        dis_pos_average = dis_pos_all / (i + 1)
                        dis_neg_average = dis_neg_all / (i + 1)

                        ############test############
                        batches = int((test_size * 4) / batch_size)
                        right = 0.
                        for ii in range(batches):
                            answer, result_test, result_state_pos = sess.run([s_pos, test1, state_pos],
                                              feed_dict={image_pos: test_image[ii * batch_size:(ii + 1) * batch_size],
                                                         graph_pos: test_graph[ii * batch_size:(ii + 1) * batch_size]}
                                              )
                            answer = np.asarray(answer)

                            for j in range(batch_size / 4):
                                a = []
                                for k in range(j * 4, (j + 1) * 4):
                                    a.append(answer[k][0])
                                if np.argmax(a) == 0:
                                    right += 1.

                        print(answer)
                        # print("result_state(0 row)")
                        # print(result_state_pos[0])
                        # print("result_test(0 row)")
                        # print(result_test[0])
                        # print("graph_nodes")
                        # for graph_ in test_graph[ii * batch_size:(ii + 1) * batch_size]:
                        #     print (look_enable_node(graph_))

                        accurancy = float(right / test_size)

                        if accurancy > best_accurancy:
                            best_accurancy = accurancy
                            best_epoch = epoch
                            # saver.save(sess, "trained_model/cm_ggnn.ckpt")

                        print('now():' + str(datetime.now()))
                        print("Train Epoch:", '%d' % epoch, "Batch:", '%d' % i,
                              "total cost:", "{:.9f}".format(c_average), "pred score distance:", "{:.9f}".format(score_average),
                              "parameter cost:", "{:.9f}".format(para_average), "postive score:", "{:.9f}".format(dis_pos_average),
                              "negative score:", "{:.9f}".format(dis_neg_average),
                              "accurancy:", ".{:.9f}".format(accurancy))
                        print("Epoch:", '%d' % epoch, "finished", "Best accurancy: %f" % best_accurancy,
                              "Best epoch: %d" % best_epoch)
                        print("batch_size: %d, hidden_size: %d, n_steps: %d, learning_rate: %f" % (
                            batch_size, hidden_size, n_steps, learning_rate))

                else:
                    no_count += 1

            c_average = c_all / train_batch
            score_average = score_all / train_batch
            para_average = para_all / train_batch
            dis_pos_average = dis_pos_all / train_batch
            dis_neg_average = dis_neg_all / train_batch

            print("Train Epoch:", '%d' % epoch, "finished",
                  "total cost:", "{:.9f}".format(c_average), "pred score distance:", "{:.9f}".format(score_average),
                  "parameter cost:", "{:.9f}".format(para_average),
                  "postive score:", "{:.9f}".format(dis_pos_average), "negative score:", "{:.9f}".format(dis_neg_average))
            print("Epoch:", '%d' % epoch, "finished", "Best accurancy: %f" % best_accurancy,
                  "Best epoch: %d" % best_epoch)
            print("batch_size: %d, hidden_size: %d, n_steps: %d, learning_rate: %f" % (
                batch_size, hidden_size, n_steps, learning_rate))

            ############test############
            batches = int((test_size*4) / batch_size)
            right = 0.
            for i in range(batches):
                answer, result_state_pos = sess.run([s_pos, state_pos],
                                  feed_dict={image_pos: test_image[i * batch_size:(i + 1) * batch_size],
                                             graph_pos: test_graph[i * batch_size:(i + 1) * batch_size]}
                                  )
                answer = np.asarray(answer)

                for j in range(batch_size / 4):
                    a = []
                    for k in range(j * 4, (j + 1) * 4):
                        a.append(answer[k][0])
                    if np.argmax(a) == 0:
                        right += 1.
            print (answer)
            # print("result_state_pos")
            # # print(result_state_pos)
            accurancy = float(right / test_size)
            print("Test Epoch:", '%d' % epoch, "accuracy:", "{:.9f}".format(accurancy))

            if accurancy > best_accurancy:
                best_accurancy = accurancy
                best_epoch = epoch
                saver.save(sess, "trained_model/cm_ggnn.ckpt")

            print('now():' + str(datetime.now()))
            print("batch_size: %d, hidden_size: %d, n_steps: %d, learning_rate: %f" % (
            batch_size, hidden_size, n_steps, learning_rate))
            print("Epoch:", '%d' % epoch, "finished", "Best accurancy: %f" % best_accurancy,
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
    for batch_size in [16]: #### n*8
        for hidden_size in [16, 64]:
            for n_steps in [1, 1]:
                for learning_rate in [0.001]:
                    for opt in ['RMSProp', 'Adam']:
                        for beta in [0.0001, 0.001]:
                            accurancy = cm_ggnn(batch_size, hidden_size, n_steps, learning_rate, G, num_category, opt, i, beta)
                            if accurancy > best_accurancy:
                                best_accurancy = accurancy
                                best_parameter = [batch_size, hidden_size, n_steps, learning_rate]
                                print("best parameter is batch_size, hidden_size, n_steps, learning_rate, optimizer:%d, %d ,%d ,%f, %s" % (batch_size,
                                                                                                                        hidden_size, n_steps, learning_rate, opt))
                                i += 1


