#!/usr/bin/env python
import pdb

from c45 import C45

# c1 = C45("../data/iris/iris.data", "../data/iris/iris.names")
#c1 = C45("../data/weather/weather.data", "../data/weather/weather.names")
# c1 = C45("../data/breast-cancer/breast-cancer-new.data", "../data/breast-cancer/breast-cancer.names")
# c1 = C45("../data/cheat/cheat.data", "../data/cheat/cheat.names")
c1 = C45("../data/car/car.data", "../data/car/car.c45-names")
data = c1.fetchData()
c1.preprocessData()
tree = c1.generateTree()
c1.printTree()
error = c1.validate(tree, data)
print('traning-error: ' + str(1-error/len(data)))
print('pruned-tree')
prunedTree = c1.pruneTheTree()
error = c1.validate(prunedTree, data)
print('pruned-tree-traning-error: ' + str(1-error/len(data)))