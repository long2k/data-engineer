#!/usr/bin/env python
import pdb

from c45 import C45

# c1 = C45("../data/iris/iris.data", "../data/iris/iris.names")
#c1 = C45("../data/weather/weather.data", "../data/weather/weather.names")
c1 = C45("../data/breast-cancer/breast-cancer-new.data", "../data/breast-cancer/breast-cancer.names")
# c1 = C45("../data/cheat/cheat.data", "../data/cheat/cheat.names")
data = c1.fetchData()
c1.preprocessData()
tree = c1.generateTree()
#c1.printTree()
error = c1.validate(tree, data)
print(1-error/len(data))
c1.pruneTheTree()