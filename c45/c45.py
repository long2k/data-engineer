import math


class C45:
    """Creates a decision tree with C4.5 algorithm"""

    def __init__(self, pathToData, pathToNames):
        self.filePathToData = pathToData
        self.filePathToNames = pathToNames
        self.data = []  # du lieu duoc nap
        self.classes = []  # lop phan loai
        self.numAttributes = -1  # so thuoc tinh
        self.attrValues = {}  # gia tri thuoc tinh
        self.attributes = []  # thuoc tinh
        self.tree = None  # cay
        self.prunedTree = None
        self.testingData = []
        self.minNumberOfInstances = 0

    def getTestingData(self, testingFilePath):
        with open(testingFilePath, "r") as file:
            for line in file:
                row = [x.strip() for x in line.split(",")]
                if row != [] or row != [""]:
                    self.testingData.append(row)

    def fetchData(self):
        with open(self.filePathToNames, "r") as file:
            classes = file.readline()
            self.classes = [x.strip() for x in classes.split(",")]
            # add attributes
            for line in file:
                [attribute, values] = [x.strip() for x in line.split(":")]
                values = [x.strip() for x in values.split(",")]
                self.attrValues[attribute] = values
        self.numAttributes = len(self.attrValues.keys())
        self.attributes = list(self.attrValues.keys())
        with open(self.filePathToData, "r") as file:
            for line in file:
                row = [x.strip() for x in line.split(",")]
                if row != [] or row != [""]:
                    self.data.append(row)
        return self.data

    def preprocessData(self):
        for index, row in enumerate(self.data):
            for attr_index in range(self.numAttributes):
                # du lieu lien tuc thi ep kieu string => float
                if (not self.isAttrDiscrete(self.attributes[attr_index])):
                    self.data[index][attr_index] = float(self.data[index][attr_index])

    def preprocessTestingData(self):
        for idx, row in enumerate(self.testingData):
            for attr_idx in range(self.numAttributes):
                if self.isAttrDiscrete(self.attributes[attr_idx]):
                    self.testingData[idx][attr_idx] = float(self.testingData[idx][attr_idx])

    def predict(self, node, dataRow):
        if not node.isLeaf:
            # get node attribute index
            nodeAttrIdx = -1
            for attrIdx in range(len(self.attributes)):
                if self.attributes[attrIdx] == node.label:
                    nodeAttrIdx = attrIdx
                    break
            if node.threshold is None:
                for idx, child in enumerate(node.children):
                    if dataRow[nodeAttrIdx] == self.attrValues[node.label][idx]:
                        if child.isLeaf:
                            return str(child.label)
                        else:
                            return self.predict(child, dataRow)
            else:
                leftChild = node.children[0]
                rightChild = node.children[1]

                if float(dataRow[nodeAttrIdx]) <= node.threshold:
                    if leftChild.isLeaf:
                        return str(leftChild.label)
                    else:
                        return self.predict(leftChild, dataRow)
                else:
                    if rightChild.isLeaf:
                        return str(rightChild.label)
                    else:
                        return self.predict(rightChild, dataRow)

    def validate(self, tree, data):
        result = []
        error = 0
        for row in data:
            # print(self.predict(self.tree, row))
            result.append(self.predict(tree, row))
        for idx, r in enumerate(result):
            if data[idx][-1] != r:
                error += 1

        return error

    def printTree(self):
        self.printNode(self.tree)

    def printPrunedTree(self):
        self.printNode(self.prunedTree)

    def printNode(self, node, indent=""):
        if not node.isLeaf:
            if node.threshold is None:
                # roi rac
                for index, child in enumerate(node.children):
                    if child.isLeaf:
                        print(indent + node.label + " ID: " + str(node._id) + " = " + self.attrValues[node.label][
                            index] + " : " + child.label + ' idc: ' + str(child._id))
                    else:
                        print(
                            indent + node.label + " ID: " + str(node._id) + " = " + self.attrValues[node.label][index])
                        self.printNode(child, indent + "|   ")
            else:
                # lien tuc
                leftChild = node.children[0]
                rightChild = node.children[1]
                if leftChild.isLeaf:
                    print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
                else:
                    print(indent + node.label + " <= " + str(node.threshold))
                    self.printNode(leftChild, indent + "|    ")

                if rightChild.isLeaf:
                    print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
                else:
                    print(indent + node.label + " > " + str(node.threshold))
                    self.printNode(rightChild, indent + "|   ")

    def generateTree(self):
        self.tree = self.recursiveGenerateTree(self.data, self.attributes, self.getMajClass(self.data))
        return self.tree

    def recursiveGenerateTree(self, curData, curAttributes, parentMajClass):

        print(len(curData))
        allSameClass = self.allSameClass(curData)
        allSameAttr = self.allSameAttrValue(curData)
        nodeMajClass = self.getMajClass(curData)
        if len(curData) == 0:
            # Fail
            return Node(True, parentMajClass, None, nodeMajClass, None)
        elif allSameClass is not False:
            # return a node with that class
            return Node(True, allSameClass, None, nodeMajClass,None)
        elif allSameAttr is True:
            return Node(True, parentMajClass, None, nodeMajClass,None)
        elif len(curAttributes) == 0:
            # return a node with the majority class
            return Node(True, nodeMajClass, None, nodeMajClass,None)
        else:
            # chon thuoc tinh, xoa thuoc tinh tot nhat trong danh sach thuoc tinh
            # print(i)
            (best, best_threshold, splitted) = self.splitAttribute(curData, curAttributes)

            remainingAttributes = curAttributes[:]
            remainingAttributes.remove(best)
            node = Node(False, best, best_threshold, nodeMajClass,None)


            countSubset = 0
            for subset in splitted:
                if len(subset) >= self.minNumberOfInstances:
                    countSubset += 1
            if countSubset < 2:
                return Node(True, parentMajClass, None, nodeMajClass,None)
            node.children = [self.recursiveGenerateTree(subset, remainingAttributes, nodeMajClass) for subset
                             in splitted]
            node.subsetData = curData
            return node

    def getMajClass(self, curData):
        freq = [0] * len(self.classes)  # tao arr freq[so luong class]
        for row in curData:
            index = self.classes.index(row[-1])
            freq[index] += 1
        maxInd = freq.index(max(freq))
        return self.classes[maxInd]

    def allSameAttrValue(self, data):
        stop = True
        for row in data:
            for idx in range(len(row) - 1):
                if data[0][idx] != row[idx]:
                    stop = False
                    return stop
        return stop

    # kiem tra xem du lieu deu chung 1 class khong row[-1] = classname
    def allSameClass(self, data):
        if len(data) == 0:
            return False
        for row in data:
            # so sanh voi class cua data dau tien
            if row[-1] != data[0][-1]:
                return False
        return data[0][-1]

    def isAttrDiscrete(self, attribute):
        if attribute not in self.attributes:
            raise ValueError("Attribute not listed")
        elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
            return False
        else:
            return True

    def splitAttribute(self, curData, curAttributes):
        splitted = []
        maxEnt = 0  # * float("inf")  # -INF entropy luon + ?
        best_attribute = -1
        # None for discrete attributes, threshold value for continuous attributes
        best_threshold = None
        for attr_idx, attribute in enumerate(curAttributes):
            indexOfAttribute = self.attributes.index(attribute)
            if self.isAttrDiscrete(attribute):
                # split curData into n-subsets, where n is the number of
                # different values of attribute i. Choose the attribute with
                # the max gain
                valuesForAttribute = self.attrValues[attribute]
                subsets = [[] for a in valuesForAttribute]  # thống kê các mẫu có thuộc tính chứa giá trị này
                for row in curData:
                    for index in range(len(valuesForAttribute)):
                        if row[indexOfAttribute] == valuesForAttribute[index]:  # index ?
                            subsets[index].append(row)
                            break
                e = self.gain(curData, subsets)
                if e >= maxEnt:
                    maxEnt = e
                    splitted = subsets
                    best_attribute = attribute
                    best_threshold = None
            else:
                # sort the data according to the column.Then try all
                # possible adjacent pairs. Choose the one that
                # yields maximum gain
                curData.sort(key=lambda x: x[indexOfAttribute])
                for j in range(0, len(curData) - 1):
                    if curData[j][indexOfAttribute] != curData[j + 1][indexOfAttribute]:
                        threshold = (curData[j][indexOfAttribute] + curData[j + 1][
                            indexOfAttribute]) / 2  # gia tri trung binh 2 row lien ke
                        less = []  # tap con co gia tri thuoc tinh < threshold
                        greater = []
                        for row in curData:
                            if (row[indexOfAttribute] > threshold):
                                greater.append(row)
                            else:
                                less.append(row)
                        e = self.gain(curData, [less, greater])  # test do do thong tin khi chia nguong
                        if e >= maxEnt:
                            splitted = [less, greater]
                            maxEnt = e
                            best_attribute = attribute
                            best_threshold = threshold
        #     print(attribute + str(len(curData)) + ' ' + str(e))
        # print('select ' + best_attribute)
        return (best_attribute, best_threshold, splitted)

    # do tang thong tin khi chia thanh cac tap con
    def gain(self, unionSet, subsets):  # gain split
        # input : data and disjoint subsets of it
        # output : information gain
        S = len(unionSet)
        # calculate impurity before split
        impurityBeforeSplit = self.entropy(unionSet)  # parent node entropy
        # calculate impurity after split
        weights = [len(subset) / S for subset in subsets]
        impurityAfterSplit = 0  # child nodes entropy
        for i in range(len(subsets)):
            impurityAfterSplit += weights[i] * self.entropy(subsets[i])
        # calculate total gain
        totalGain = impurityBeforeSplit - impurityAfterSplit
        # split info
        splitInfo = self.split_info(subsets, S)
        if totalGain == 0:
            return 0

        gainRatio = totalGain / splitInfo
        return gainRatio

    def split_info(self, subsets, total_set):
        split_info = 0
        for subset in subsets:
            split_info += - (len(subset) / total_set) * self.log(len(subset) / total_set)
        return split_info

    def entropy(self, dataSet):
        S = len(dataSet)
        if S == 0:
            return 0
        num_classes = [0 for i in self.classes]
        # dem so class xuat hien trong dataset
        for row in dataSet:
            classIndex = list(self.classes).index(row[-1])
            num_classes[classIndex] += 1
        # chia S ra tan so xuat hien cua lop do trong data hien tai
        num_classes = [x / S for x in num_classes]
        ent = 0
        # tinh entropy
        for num in num_classes:
            ent += num * self.log(num)
        return ent * -1

    # logarith co so 2
    def log(self, x):
        if x == 0:
            return 0
        else:
            return math.log(x, 2)

    def getPessimisticErrBefore(self, tree, data):
        error = (self.validate(tree, data))/len(data)
        #print(error)
        return error

    def getPessimisticErrAfter(self, tree, data, numberOfLeafs):
        error = (self.validate(tree, data) + numberOfLeafs * 0.5) /len(data)
        #print(error)
        return error

    def pruneTheTree(self):
        self.prunedTree = self.tree
        self.prune(self.prunedTree)
        #print(self.prunedTree.children)
        self.printPrunedTree()
        return self.prunedTree

    def prune(self, node):

        if not node.isLeaf:
            for child in node.children:
                self.prune(child)
            # if node._id == 0:
            #     return
            after = self.getPessimisticErrAfter(self.prunedTree, node.subsetData, len(node.children))
            # la node sat nhat voi leaf
            tmpChildren = node.children
            tmpLabel = node.label
            # print(str(node.isLeaf) + ' ' + tmpLabel + ' ' + str(node._id))
            node.isLeaf = True
            node.children = None
            node.label = node.majClass
            before = self.getPessimisticErrBefore(self.prunedTree, node.subsetData)
            # for c in tmpChildren:
            #     print(c.label)
            #self.printPrunedTree()
            # print(before)
            # print(after)
            #neu khong can tia
            if before > after:
                print(before - after)
                node.children = tmpChildren
                node.isLeaf = False
                node.label = tmpLabel
        return

    # def getNode(self, node, nodeIdToSplit, newNode):
    #     newNode = node
    #     for child in node:
    #         newNode
    #         self.getNode(child, nodeIdToSplit, newTree)
    #     if node._id == nodeIdToSplit:
    #         return child

# gom nhan, nguong(neu co), la nut la, cac nut con
class Node:
    _id = 0

    def __init__(self, isLeaf, label, threshold, majClass, data):
        self._id = Node._id
        Node._id += 1
        self.label = label
        self.threshold = threshold
        self.isLeaf = isLeaf
        self.children = []
        self.majClass = majClass  # class xuat hien nhieu nhat trong subset tai node do
        self.subsetData = data
