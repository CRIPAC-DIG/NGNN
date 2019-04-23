import numpy as np
import json
import pickle
import random
# image_x = tf.placeholder(tf.float32, [batch_size, num_category, 2048])
# image_y = tf.placeholder(tf.float32, [batch_size, 2048])
# category_y = tf.placeholder(tf.int32, [batch_size])
# grah = tf.placeholder(tf.float32, [batch_size, num_category, num_category])

# fc = open('category_summarize_1000.json', 'r')
# dict_list = json.load(fc)

fic = open('category_summarize_1000.json', 'r')
category_item = json.load(fic)
num_category = len(category_item)
print('more than 1000 category: %d' % num_category)
fr = open('cid2rcid_1000.json', 'r')
cid2rcid = json.load(fr)
#print (cid2rcid)
feature_path = '/home/lizekun/polyvore_image_vectors/'
#feature_path = '/home/cuizeyu/polyvore_image_vectors/'

total_graph = np.zeros((num_category, num_category), dtype=np.float32)
# summraize the total_graph first
ftrain = open('train_no_dup_new_1000.json', 'r')
outfit_list = json.load(ftrain)
ftrain.close()

for outfit in outfit_list:
    cate_list = outfit['items_category']
    for cid in cate_list:
        for cjd in cate_list:
            if cid != cjd:
                rcid = int(cid2rcid[str(cid)])
                rcjd = int(cid2rcid[str(cjd)])
                total_graph[rcid][rcjd] += 1.
                total_graph[rcjd][rcid] += 1.

for idx in range(num_category):
    frequence = category_item[idx]['frequency']
    total_graph[idx] = total_graph[idx] / frequence

# print total_graph

per_outfit = 8

def load_train_data(i, batch_size):
    ftrain = open('train_no_dup_new_1000.json', 'r')
    outfit_list = json.load(ftrain)
    size = len(outfit_list)
    size_ = size * per_outfit
    time = int(batch_size / per_outfit)
    i = i * time

    image_pos = np.zeros((batch_size, num_category, 2048), dtype=np.float32)
    image_neg = np.zeros((batch_size, num_category, 2048), dtype=np.float32)
    graph_pos = np.zeros((batch_size, num_category, num_category), dtype=np.float32)
    graph_neg = np.zeros((batch_size, num_category, num_category), dtype=np.float32)
    now_size = 0

    for outfit in outfit_list[i : i + time]:
        ii = outfit['items_index']
        ci = outfit['items_category']
        sid = outfit['set_id']
        len_ = len(ii)
        j_list = []
        for j in range(len_):
            j_list.append(j)
        for j in range(per_outfit - len_):
            j_list.append(random.randint(0, len_ - 1))
        for j in j_list:
            list_ = []
            for k in range(len(ii)):
                cid = ci[k]
                iid = ii[k]
                rcid = int(cid2rcid[str(cid)])
                feature = json.load(open(feature_path + str(sid)+ '_' + str(iid)+ '.json'))
                if k == j:  #if k==j k is y,else k is x
                    image_pos[now_size][rcid] = feature
                    rcid_pos = rcid
                    # for a in category_item:
                    #     if a['id'] == cid:
                    #         i = a
                    #         break
                    i = random.choice(category_item)
                    rcid_neg = cid2rcid[str(i['id'])]
                    image_neg[now_size][rcid_neg] = json.load(open(feature_path + random.choice(i['items']) + '.json'))

                else:
                    image_neg[now_size][rcid] = feature
                    image_pos[now_size][rcid] = feature
                    list_.append(rcid)

            list_pos = list_[:]
            list_pos.append(rcid_pos)
            for a in list_pos:
                for b in list_pos:
                    if b != a:
                        graph_pos[now_size][a][b] = 1.  # total_graph[a][b]
                        # total_graph[a][b] = 1.
            list_neg = list_[:]
            list_neg.append(rcid_neg)
            for a in list_neg:
                for b in list_neg:
                    if b != a:
                        graph_neg[now_size][a][b] = 1.  # total_graph[a][b]

            now_size += 1

    graph_pos = reuniform_graph(graph_pos)
    graph_neg = reuniform_graph(graph_neg)

    return image_pos, image_neg, graph_pos, graph_neg, size_


def load_train_size():
    ftrain = open('train_no_dup_new_1000.json', 'r')
    train_list = json.load(ftrain)
    train_size = len(train_list)
    train_size_ = per_outfit * train_size
    return train_size, train_size_


def load_graph():
    return total_graph


def load_num_category():
    return num_category

# def load_valid_data():
#     fvalid = open('valid_no_dup_new_1000.json', 'r')
#     valid_list = json.load(fvalid)
#     valid_size = len(valid_list)
#     valid_size_ = 0
#     for i in range(valid_size):
#         outfit = valid_list[i]
#         ii = outfit['items_index']
#         valid_size_ += len(ii)
#     image_x, image_pos, image_neg, category_pos, category_neg, graph = load_data(valid_size, valid_size_, valid_list)
#     return image_x, image_pos, image_neg, category_pos, category_neg, graph, valid_size_
#
# def load_test_data():
#     ftest = open('test_no_dup_new_1000.json', 'r')
#     test_list = json.load(ftest)
#     test_size = len(test_list)
#     test_size_ = 0
#     for i in range(test_size):
#         outfit = test_list[i]
#         ii = outfit['items_index']
#         test_size_ += len(ii)
#     image_x, image_pos, image_neg, category_pos, category_neg, graph = load_data(test_size, test_size_, test_list)
#     return image_x, image_pos, image_neg, category_pos, category_neg, graph, test_size_


def reuniform_graph(graph):
    # graph_size = (size_, num_category, num_category)
    size_ = len(graph)
    num_category = len(graph[0])
    graph = graph.astype(np.float32)
    for idx in range(size_):
        sum_g = np.sum(graph[idx], axis=0)
        for jdx in range(num_category):
            if sum_g[jdx] != 0:
                tempt = graph[idx][jdx] / float(sum_g[jdx])
                graph[idx][jdx] = tempt

    return graph

#
# if __name__ == "__main__":
#     temp_graph = np.asarray([[[0,1,1,0], [1,0,1,0], [1,1,0,0], [0,0,0,0]]])
#     reuniform_graph(temp_graph)
#     load_train_data(2, 16)
