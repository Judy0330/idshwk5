import math
from sklearn.ensemble import RandomForestClassifier
import os

# import numpy as np

domainlist = []


def calEntropy(name):
    h = 0.0
    sumt = 0
    letter = [0] * 26
    name = name.lower()
    for c in name:
        if c.isalpha():
            letter[ord(c) - ord('a')] += 1
            sumt += 1
    # print('\n', letter)
    for i in range(26):
        p = 1.0 * letter[i] / sumt
        if p > 0:
            h += -(p * math.log(p, 2))
    return h


class Domain:
    # def __init__(self, _name, _label, _min, _max, _numip, _ipset):
    #     self.name = _name
    #     self.label = _label
    #     self.ttlmin = _min
    #     self.ttlmax = _max
    #     self.numip = _numip
    #     self.ipset = _ipset

    def __init__(self, _name, _label, _name_len, _name_num_cnt, _entropy):
        self.name = _name
        self.label = _label
        self.name_len = _name_len
        self.name_num_cnt = _name_num_cnt
        self.entropy = _entropy

    # def returnData(self):
    #     return [self.ttlmin, self.ttlmax, self.numip]

    def returnData(self):
        return [self.name_len, self.name_num_cnt, self.entropy]

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1


def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            name_len = len(tokens[0])
            name_num_cnt = sum(c.isnumeric() for c in name)
            entropy = calEntropy(name)
            # ttlmin = int(tokens[2])
            # ttlmax = int(tokens[3])
            # numIP = int(tokens[4])
            # ipset = set()
            # for i in range(numIP):
            #     ipset.add(tokens[5 + i])
            # domainlist.append(Domain(name, label, ttlmin, ttlmax, numIP, ipset))
            domainlist.append(Domain(name, label, name_len, name_num_cnt, entropy))


def main():
    print("Initialize Raw Objects")
    # initData("baddomaininfo")
    # initData("gooddomaininfo")
    initData("train.txt")
    featureMatrix = []
    labelList = []
    print("Initialize Matrix")
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())
    print(featureMatrix)
    print("Begin Training")
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    print("Begin Predicting")
    # print(clf.predict([[3600, 10000, 3]]))
    # print(clf.predict([[3600, 3600, 2]]))
    # print(clf.predict([[100, 100, 3]]))
    # print(clf.predict([[100, 100, 1]]))
    with open("test.txt") as r, open("result.txt", "w") as w:
        for line in r:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tmp = [len(line), sum(c.isnumeric() for c in line), calEntropy(line)]
            w.writelines(clf.predict(tmp))
    r.close()
    w.close()


if __name__ == '__main__':
    main()
    os.system("open /Applications/Calculator.app")  # ?
