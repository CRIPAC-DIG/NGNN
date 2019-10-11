import numpy as np
import json
import pickle
import random
# image_x = tf.placeholder(tf.float32, [batch_size, num_category, 2048])
# image_y = tf.placeholder(tf.float32, [batch_size, 2048])
# category_y = tf.placeholder(tf.int32, [batch_size])
# grah = tf.placeholder(tf.float32, [batch_size, num_category, num_category])

# fc = open('category_summarize_100.json', 'r')
# dict_list = json.load(fc)

fic = open('category_summarize_100.json', 'r')
category_item = json.load(fic)
num_category = len(category_item)
print('more than 100 category: %d' % num_category)
fr = open('cid2rcid_100.json', 'r')
cid2rcid = json.load(fr)
#print (cid2rcid)
image_feature_path = '/home/lizekun/polyvore_image_vectors/'
#image_feature_path = '/home/cuizeyu/polyvore_image_vectors/'
text_feature_path = '/home/lizekun/polyvore_text_onehot_vectors/'
#text_feature_path = '/home/cuizeyu/polyvore_text_vectors/'

total_graph = np.zeros((num_category, num_category), dtype=np.float32)
per_outfit = 8

def load_graph():
    return total_graph

def load_num_category():
    return num_category

def load_train_data(i, batch_size, outfit_list):

    size_ = batch_size
    time = int(batch_size / per_outfit)
    i = i * time

    image_pos = np.zeros((size_, num_category, 2048), dtype=np.float32)
    image_neg = np.zeros((size_, num_category, 2048), dtype=np.float32)
    text_pos = np.zeros((size_, num_category, 2757), dtype=np.float32)
    text_neg = np.zeros((size_, num_category, 2757), dtype=np.float32)
    graph_pos = np.zeros((size_, num_category, num_category), dtype=np.float32)
    graph_neg = np.zeros((size_, num_category, num_category), dtype=np.float32)
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
                image_feature = json.load(open(image_feature_path + str(sid)+ '_' + str(iid)+ '.json'))
                text_feature = json.load(open(text_feature_path + str(sid) + '_' + str(iid) + '.json'))
                if k == j:  #if k==j k is y,else k is x
                    image_pos[now_size][rcid] = image_feature
                    text_pos[now_size][rcid] = text_feature
                    rcid_pos = rcid
                    # for a in category_item:
                    #     if a['id'] == cid:
                    #         i = a
                    #         break
                    i = random.choice(category_item)
                    rcid_neg = cid2rcid[str(i['id'])]
                    ri = random.choice(i['items'])
                    image_neg[now_size][rcid_neg] = json.load(open(image_feature_path + ri + '.json'))
                    text_neg[now_size][rcid_neg] = json.load(open(text_feature_path + ri + '.json'))

                else:
                    image_neg[now_size][rcid] = image_feature
                    image_pos[now_size][rcid] = image_feature
                    text_neg[now_size][rcid] = text_feature
                    text_pos[now_size][rcid] = text_feature
                    list_.append(rcid)

            list_pos = list_[:]
            list_pos.append(rcid_pos)
            for a in list_pos:
                for b in list_pos:
                    if b != a:
                        graph_pos[now_size][a][b] = 1.
                        total_graph[a][b] = 1.
            list_neg = list_[:]
            list_neg.append(rcid_neg)
            for a in list_neg:
                for b in list_neg:
                    if b != a:
                        graph_neg[now_size][a][b] = 1.

            now_size += 1

    return image_pos, image_neg, text_pos, text_neg, graph_pos, graph_neg, size_


def load_train_size():
    ftrain = open('train_no_dup_new_100.json', 'r')
    train_list = json.load(ftrain)
    train_size = len(train_list)
    train_size_ = per_outfit * train_size
    return train_size, train_size_


def load_fitb_data(index, batch_size, outfit_list):

    time = int(batch_size / 4)
    num_category = load_num_category()

    image = np.zeros((batch_size, num_category, 2048), dtype=np.float32)
    text = np.zeros((batch_size, num_category, 2757), dtype=np.float32)
    graph = np.zeros((batch_size, num_category, num_category), dtype=np.float32)
    outfit_list_ = outfit_list[index*time: (index + 1)*time]
    for i in range(len(outfit_list_)):
        outfit = outfit_list_[i]
        ii = outfit['items_index']
        ci = outfit['items_category']
        sid = outfit['set_id']
        rcid_list = []
        length = len(ii)
        blank_index = random.randint(0, length - 1)
        for j in range(len(ii)):
            cid = ci[j]
            iid = ii[j]
            rcid = int(cid2rcid[str(cid)])
            image_feature = json.load(open(image_feature_path + str(sid) + '_' + str(iid) + '.json'))
            text_feature = json.load(open(text_feature_path + str(sid) + '_' + str(iid) + '.json'))
            if j == blank_index:
                rcid_pos = rcid
                image[i * 4][rcid] = image_feature
                text[i * 4][rcid] = text_feature
                # r1 = random.choice(category_item)
                # feature_w1[rcid] = json.load(open(feature_path + random.choice(r1['items']) + '.json'))
                # category_y[i * 4 + 1] = rcid
                # r2 = random.choice(category_item)
                # feature_w2[rcid] = json.load(open(feature_path + random.choice(r2['items']) + '.json'))
                # category_y[i * 4 + 2] = rcid
                # r3 = random.choice(category_item)
                # feature_w3[rcid] = json.load(open(feature_path + random.choice(r3['items']) + '.json'))
                # category_y[i * 4 + 3] = rcid
                r1 = random.choice(category_item)
                rcid_w1 = cid2rcid[str(r1['id'])]
                r11 = random.choice(r1['items'])
                image[i * 4 + 1][rcid_w1] = json.load(open(image_feature_path + r11 + '.json'))
                text[i * 4 + 1][rcid_w1] = json.load(open(text_feature_path + r11 + '.json'))
                r2 = random.choice(category_item)
                rcid_w2 = cid2rcid[str(r2['id'])]
                r22 = random.choice(r2['items'])
                image[i * 4 + 2][rcid_w2] = json.load(open(image_feature_path + r22 + '.json'))
                text[i * 4 + 2][rcid_w2] = json.load(open(text_feature_path + r22 + '.json'))
                r3 = random.choice(category_item)
                rcid_w3 = cid2rcid[str(r3['id'])]
                r33 = random.choice(r3['items'])
                image[i * 4 + 3][rcid_w3] = json.load(open(image_feature_path + r33 + '.json'))
                text[i * 4 + 3][rcid_w3] = json.load(open(text_feature_path + r33 + '.json'))

            else:
                rcid_list.append(rcid)
                image[i * 4][rcid] = image_feature
                image[i * 4 + 1][rcid] = image_feature
                image[i * 4 + 2][rcid] = image_feature
                image[i * 4 + 3][rcid] = image_feature
                text[i * 4][rcid] = text_feature
                text[i * 4 + 1][rcid] = text_feature
                text[i * 4 + 2][rcid] = text_feature
                text[i * 4 + 3][rcid] = text_feature

        rl_pos = rcid_list[:]
        rl_pos.append(rcid_pos)
        g1 = np.zeros((num_category, num_category), dtype=np.float32)
        for a in rl_pos:
            for b in rl_pos:
                if b != a:
                    g1[a][b] = 1.
        graph[i * 4] = g1

        rl_w1 = rcid_list[:]
        rl_w1.append(rcid_w1)
        g2 = np.zeros((num_category, num_category), dtype=np.float32)
        for a in rl_w1:
            for b in rl_w1:
                if b != a:
                    g2[a][b] = 1.
        graph[i * 4 + 1] = g2

        rl_w2 = rcid_list[:]
        rl_w2.append(rcid_w2)
        g3 = np.zeros((num_category, num_category), dtype=np.float32)
        for a in rl_w2:
            for b in rl_w2:
                if b != a:
                    g3[a][b] = 1.
        graph[i * 4 + 2] = g3

        rl_w3 = rcid_list[:]
        rl_w3.append(rcid_w3)
        g4 = np.zeros((num_category, num_category), dtype=np.float32)
        for a in rl_w3:
            for b in rl_w3:
                if b != a:
                    g4[a][b] = 1.
        graph[i * 4 + 3] = g4

    return image, text, graph

def load_test_size():
    ftest = open('test_no_dup_new_100.json', 'r')
    test_list = json.load(ftest)
    test_size = len(test_list)

    return test_size



def load_auc_data(index, batch_size, outfit_list):

    time = int(batch_size / 2)

    num_category = load_num_category()

    image = np.zeros((batch_size, num_category, 2048), dtype=np.float32)
    text = np.zeros((batch_size, num_category, 2757), dtype=np.float32)
    graph = np.zeros((batch_size, num_category, num_category), dtype=np.float32)
    outfit_list_ = outfit_list[index*time: (index + 1)*time]
    for i in range(len(outfit_list_)):
        outfit = outfit_list_[i]
        ii = outfit['items_index']
        ci = outfit['items_category']
        sid = outfit['set_id']
        rcid_list = []
        length = len(ii)
        graph_pos = []
        graph_neg = []
        for j in range(length):
            cid = ci[j]
            iid = ii[j]
            rcid = int(cid2rcid[str(cid)])
            image_feature = json.load(open(image_feature_path + str(sid) + '_' + str(iid) + '.json'))
            image[i * 2][rcid] = image_feature
            text_feature = json.load(open(text_feature_path + str(sid) + '_' + str(iid) + '.json'))
            text[i * 2][rcid] = text_feature
            graph_pos.append(rcid)
        for j in range(length):
            r = random.choice(category_item)
            rcid_w = cid2rcid[str(r['id'])]
            rr = random.choice(r['items'])
            image[i * 2 + 1][rcid_w] = json.load(open(image_feature_path + rr + '.json'))
            text[i * 2 + 1][rcid_w] = json.load(open(text_feature_path + rr + '.json'))
            graph_neg.append(rcid_w)

        g = np.zeros((num_category, num_category), dtype=np.float32)
        for a in graph_pos:
            for b in graph_pos:
                if b != a:
                    g[a][b] = 1.
        graph[i * 2] = g

        g = np.zeros((num_category, num_category), dtype=np.float32)
        for a in graph_neg:
            for b in graph_neg:
                if b != a:
                    g[a][b] = 1.
        graph[i * 2 + 1] = g

    return image, text, graph



# def load_valid_data():
#     fvalid = open('valid_no_dup_new_100.json', 'r')
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
#     ftest = open('test_no_dup_new_100.json', 'r')
#     test_list = json.load(ftest)
#     test_size = len(test_list)
#     test_size_ = 0
#     for i in range(test_size):
#         outfit = test_list[i]
#         ii = outfit['items_index']
#         test_size_ += len(ii)
#     image_x, image_pos, image_neg, category_pos, category_neg, graph = load_data(test_size, test_size_, test_list)
#     return image_x, image_pos, image_neg, category_pos, category_neg, graph, test_size_




