# from c45 import C45

# c1 = C45("../data/breast-cancer/breast-cancer-new.data", "../data/breast-cancer/breast-cancer.names")
from c45 import C45

ratio = 0.8  # thay cái này 0.6 0.7 0.8...

# c45 = C45("../data/car/" + str(ratio) + "/training/car.data",
#           "../data/car/" + str(ratio) + "/testing/car.data",
#           "../data/car/1.0/car.c45-names")
# Sử dụng training làm test
c45 = C45("../data/car/1.0/car.data",
          "../data/car/1.0/car.data",
          "../data/car/1.0/car.c45-names")

data, testing_data = c45.get_data()
tree = c45.generateTree()
c45.printTree(tree)
c45.test(tree, testing_data)
print('\n ---------pruned-tree--------- \n')
prunedTree = c45.pruneTheTree()
c45.test(prunedTree, testing_data)
