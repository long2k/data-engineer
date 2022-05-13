from Node import Node
from cal_gain_ratio import gain_ratio
from read_data import get_data, get_names


class C45:

    def __init__(self, training_path, testing_path, names_path):
        # Đường dẫn
        self.training_path = training_path
        self.testing_path = testing_path
        self.names_path = names_path

        self.classes = []  # Lớp phân loại
        self.attr_values = {}  # Giá trị các thuộc tính
        self.attributes = []  # Thuộc tính

        self.tree = None  # Cây
        self.prunedTree = None  # Cây được tỉa

        self.count_leaf = 0
        self.count_node = 0

        self.matrix = {}
        # Dữ liệu
        self.training_data = []
        self.testing_data = []

        self.minNumberOfInstances = 2

    def get_data(self):
        (self.attributes, self.attr_values, self.classes) = get_names(self.names_path)
        self.training_data = self.preprocess(get_data(self.training_path))
        self.testing_data = self.preprocess(get_data(self.testing_path))
        return self.training_data, self.testing_data

    def preprocess(self, data):
        for idx, row in enumerate(data):
            for attr_idx in range(len(self.attributes)):
                # du lieu lien tuc thi ep kieu string => float
                if self.is_continuous_attr(self.attributes[attr_idx]):
                    data[idx][attr_idx] = float(data[idx][attr_idx])
        return data

    def is_continuous_attr(self, attribute):
        if len(self.attr_values[attribute]) == 1 and self.attr_values[attribute][0] == "continuous":
            return True
        else:
            return False

    def predict(self, node, data_row):
        if not node.isLeaf:
            # get node attribute index
            node_attr_idx = -1
            for attrIdx in range(len(self.attributes)):
                if self.attributes[attrIdx] == node.label:
                    node_attr_idx = attrIdx
                    break
            if node.threshold is None:
                for idx, child in enumerate(node.children):
                    if data_row[node_attr_idx] == self.attr_values[node.label][idx]:
                        if child.isLeaf:
                            return str(child.label)
                        else:
                            return self.predict(child, data_row)
            else:
                left_child = node.children[0]
                right_child = node.children[1]

                if float(data_row[node_attr_idx]) <= node.threshold:
                    if left_child.isLeaf:
                        return str(left_child.label)
                    else:
                        return self.predict(left_child, data_row)
                else:
                    if right_child.isLeaf:
                        return str(right_child.label)
                    else:
                        return self.predict(right_child, data_row)

    def test(self, tree, data):
        result = []
        error = 0
        correct = 0

        for row in data:
            result.append(self.predict(tree, row))

        for c in self.classes:
            self.matrix[c] = {x: 0 for x in self.classes}
        for idx, r in enumerate(result):
            if data[idx][-1] == r:
                self.matrix[r][r] += 1
                correct += 1
            elif data[idx][-1] != r:
                self.matrix[data[idx][-1]][r] += 1
                error += 1

        for row in self.matrix:
            print(row, end=" ")
            print(self.matrix[row])
        print('accuracy: ' + str(correct / len(data)))
        return error

    def printTree(self, tree):
        self.count_leaf = 0
        self.count_node = 0
        self.printNode(tree)
        print('number of nodes: ' + str(self.count_node))
        print('number of leaves: ' + str(self.count_leaf))

    def printNode(self, node, indent=""):
        self.count_node += 1
        if not node.isLeaf:
            if node.threshold is None:
                # roi rac
                for index, child in enumerate(node.children):
                    if child.isLeaf:
                        self.count_leaf += 1
                        self.count_node += 1
                        print(indent + node.label + " = " + self.attr_values[node.label][
                            index] + " : " + child.label)
                    else:
                        print(
                            indent + node.label + " = " + self.attr_values[node.label][index])
                        self.printNode(child, indent + "|   ")
            else:
                # lien tuc
                leftChild = node.children[0]
                rightChild = node.children[1]
                if leftChild.isLeaf:
                    self.count_leaf += 1
                    self.count_node += 1
                    print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
                else:
                    print(indent + node.label + " <= " + str(node.threshold))
                    self.printNode(leftChild, indent + "|    ")

                if rightChild.isLeaf:
                    self.count_leaf += 1
                    self.count_node += 1
                    print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
                else:
                    print(indent + node.label + " > " + str(node.threshold))
                    self.printNode(rightChild, indent + "|   ")

    def generateTree(self):
        self.tree = self.recursiveGenerateTree(self.training_data, self.attributes,
                                               self.get_major_class(self.training_data))
        return self.tree

    def recursiveGenerateTree(self, data, curAttributes, parentMajClass):

        # print(len(curData))
        allSameClass = self.allSameClass(data)
        allSameAttr = self.allSameAttrValue(data, curAttributes)
        nodeMajClass = self.get_major_class(data)
        if len(data) == 0:
            # Fail
            return Node(True, parentMajClass, None, nodeMajClass, None)
        elif allSameClass is not False:
            # return a node with that class
            return Node(True, allSameClass, None, nodeMajClass, None)
        elif allSameAttr is True:
            return Node(True, parentMajClass, None, nodeMajClass, None)
        elif len(curAttributes) == 0:
            # return a node with the majority class
            return Node(True, nodeMajClass, None, nodeMajClass, None)
        else:
            # chon thuoc tinh, xoa thuoc tinh tot nhat trong danh sach thuoc tinh
            # print(i)
            (best, best_threshold, splitted) = self.splitAttribute(data, curAttributes)

            remainingAttributes = curAttributes[:]
            remainingAttributes.remove(best)
            node = Node(False, best, best_threshold, nodeMajClass, None)

            if best_threshold is None:  # Ap dung cho gia tri roi rac
                count_subset = 0
                for subset in splitted:
                    if len(subset) >= self.minNumberOfInstances:
                        count_subset += 1
                if count_subset < 2:
                    return Node(True, parentMajClass, None, nodeMajClass, None)

            node.children = [self.recursiveGenerateTree(subset, remainingAttributes, nodeMajClass) for subset
                             in splitted]
            node.subsetData = data
            return node

    # Tìm ra class xuất hiện nhiều nhất trong tập dữ liệu
    def get_major_class(self, data):
        freq = [0] * len(self.classes)
        for row in data:
            index = self.classes.index(row[-1])
            freq[index] += 1
        max_idx = freq.index(max(freq))
        return self.classes[max_idx]

    def allSameAttrValue(self, data, attrs):
        stop = True
        for row in data:
            for a in attrs:
                attr_idx = self.attributes.index(a)
                if data[0][attr_idx] != row[attr_idx]:
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

    def splitAttribute(self, curData, curAttributes):
        splitted = []
        maxEnt = 0  # * float("inf")  # -INF entropy luon + ?
        best_attribute = -1
        # None for discrete attributes, threshold value for continuous attributes
        best_threshold = None
        for attr_idx, attribute in enumerate(curAttributes):
            indexOfAttribute = self.attributes.index(attribute)
            if not self.is_continuous_attr(attribute):
                # split curData into n-subsets, where n is the number of
                # different values of attribute i. Choose the attribute with
                # the max gain
                valuesForAttribute = self.attr_values[attribute]
                subsets = [[] for a in valuesForAttribute]  # thống kê các mẫu có thuộc tính chứa giá trị này
                for row in curData:
                    for index in range(len(valuesForAttribute)):
                        if row[indexOfAttribute] == valuesForAttribute[index]:  # index ?
                            subsets[index].append(row)
                            break
                e = gain_ratio(curData, subsets, self.classes)
                if e > maxEnt:
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
                        e = gain_ratio(curData, [less, greater], self.classes)  # test do do thong tin khi chia nguong
                        if e > maxEnt:
                            splitted = [less, greater]
                            maxEnt = e
                            best_attribute = attribute
                            best_threshold = threshold
        return best_attribute, best_threshold, splitted

    # ------------TỈA CÂY------------
    def validate(self, tree, data):
        result = []
        error = 0
        for row in data:
            result.append(self.predict(tree, row))
        for idx, r in enumerate(result):
            if data[idx][-1] != r:
                error += 1
        return error

    def getPessimisticErrBefore(self, tree, data):
        error = (self.validate(tree, data)) / len(data)
        # print(error)
        return error

    def getPessimisticErrAfter(self, tree, data, numberOfLeafs):
        error = (self.validate(tree, data) + numberOfLeafs * 0.5) / len(data)
        # print(error)
        return error

    def pruneTheTree(self):
        self.prunedTree = self.tree
        self.prune(self.prunedTree)
        self.printTree(self.prunedTree)
        return self.prunedTree

    def prune(self, node):
        if not node.isLeaf:
            for child in node.children:
                self.prune(child)
            after = self.getPessimisticErrAfter(self.prunedTree, node.subsetData, len(node.children))
            # la node sat nhat voi leaf
            tmp_children = node.children
            tmp_label = node.label
            # print(str(node.isLeaf) + ' ' + tmpLabel + ' ' + str(node._id))
            node.isLeaf = True
            node.children = None
            node.label = node.majClass
            before = self.getPessimisticErrBefore(self.prunedTree, node.subsetData)

            # neu khong can tia
            if before > after:
                # print(before - after)
                node.children = tmp_children
                node.isLeaf = False
                node.label = tmp_label
        return
