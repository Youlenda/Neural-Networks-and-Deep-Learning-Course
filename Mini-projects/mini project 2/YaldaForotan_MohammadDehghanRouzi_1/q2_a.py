import numpy as np


def data_for_training(data_s1, period1, shifting1, target_length1):
    a1 = np.arange(len(data_s1) - period1 + 1)
    data_list1 = []
    target_list1 = []
    for i in range(0, len(a1), shifting1):
        sample_list1 = []
        sample_target_list1 = []
        if i + period1 + target_length1 > len(data_s1):
            break
        for j in range(period1):
            value1 = data_s1[i + j]
            sample_list1.append(value1)
        data_list1.append(sample_list1)
        for j in range(target_length1):
            value1 = data_s1[i + j + period1]
            sample_target_list1.append(value1)
        target_list1.append(sample_target_list1)
    data_list1 = np.array(data_list1)
    target_list1 = np.array(target_list1)
    # print(data_list1)
    return data_list1, target_list1


def perpare_data(txt1):
    characters1 = sorted(list(set(txt1)))
    print('characters:', characters1)

    ''' Build a dictionary of characters '''
    char2ind1 = dict((c, i) for i, c in enumerate(characters1))
    ind2char1 = dict((i, c) for i, c in enumerate(characters1))
    # print(char2ind1)
    # print(ind2char1)
    data_s1 = np.zeros(len(txt1), dtype=int)
    for i in range(len(txt1)):
        data_s1[i] = char2ind1[txt1[i]]
    # print(data_s1)
    return data_s1, char2ind1, ind2char1


''' Read data '''
path = 'shakespeare.txt'
txt = open(path).read()


''' Build data set'''
period = 100
shifting = 5
target_length = 5

data_s, char2ind, ind2char = perpare_data(txt)
print(data_s.shape)

data_list, target_list = data_for_training(data_s, period, shifting, target_length)
print(data_list)
print(target_list)


print(data_list.shape)
print(target_list.shape)

print(data_list[0])
print(target_list[0])
print(data_list[1])
print(target_list[1])
print(data_list[2])
print(target_list[2])
print(data_list[3])
print(target_list[3])

test_ratio = 0.1
train_list = np.arange(len(data_list))
a = np.random.choice(train_list, 10, replace=False)
print(a)


