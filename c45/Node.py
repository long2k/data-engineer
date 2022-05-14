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