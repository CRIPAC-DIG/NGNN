import json
"""
filter the 
"""


f1 = open('train_no_dup.json', 'r')
train_list = json.load(f1)
f2 = open('valid_no_dup.json', 'r')
valid_list = json.load(f2)
f3 = open('test_no_dup.json', 'r')
test_list = json.load(f3)


dict_list = []
f2 = open('category_id.txt', 'r')
line = f2.readline()
while line:
    l = line.split(' ', 1)
    dict = {}
    dict['id'] = int(l[0])
    dict['name'] = l[1].rstrip("\n")
    dict['frequency'] = 0
    dict_list.append(dict)
    line = f2.readline()


for i in train_list:
    dict = {}
    item_list = i['items']
    #print (type(item_list))
    for j in item_list:
        category_id = j['categoryid']
        for k in dict_list:
            if k['id'] == category_id:
                k['frequency'] += 1

print ('total category: %d'% len(dict_list))
# with open("category_summarize.json","w") as f4:
#     json.dump(dict_list, f4)

count_100 = 0
dict_list_100 = []
cate_list_100 = []
for i in dict_list:
    if i['frequency'] >= 100:
        dict_list_100.append(i)
        cate_list_100.append(int(i['id']))
        count_100 += 1

print ('more than 100: %d'% count_100)
cate_list_100 = sorted(cate_list_100)
print (cate_list_100)
cid2rcid = {}
for i in range(len(cate_list_100)):
    cid2rcid[int(cate_list_100[i])] = i

with open("cid2rcid_100.json", "w") as f4:
    json.dump(cid2rcid, f4)

with open("category_summarize_100.json", "w") as f5:
    json.dump(dict_list_100, f5)



# # summarize the dataset information
# f1 = open('train_no_dup_new_100.json', 'r')
# train_list = json.load(f1)
# total_length = 0.
# outfit_len = len(train_list)
# for outfit in train_list:
#     total_length += len(outfit['items_index'])
# print ('total length: %d' % total_length)
#
# f1 = open('valid_no_dup_new_100.json', 'r')
# valid_list = json.load(f1)
# outfit_len = len(train_list)
# for outfit in valid_list:
#     total_length += len(outfit['items_index'])
# print ('total length: %d' % total_length)
#
# f1 = open('test_no_dup_new_100.json', 'r')
# test_list = json.load(f1)
# outfit_len = len(train_list)
# for outfit in test_list:
#     total_length += len(outfit['items_index'])
# print ('total length: %d' % total_length)
#
# print ('total length: %d' % total_length)
# #print ('train list length: %d' % outfit_len)
# print ('average: %f' % float(total_length / outfit_len))
#
#
# category_list = []
#
# for i in train_list and valid_list and test_list:
#     items = i['items']
#     for j in items:
#         category_list.append(j['categoryid'])
#
# category_list = set(category_list)
# print (category_list)
# print (len(category_list))