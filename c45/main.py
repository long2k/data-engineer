#!/usr/bin/env python
import pdb

from c45 import C45

#c1 = C45("../data/iris/iris.data", "../data/iris/iris.names")
#c1 = C45("../data/weather/weather.data", "../data/weather/weather.names")
c1 = C45("../data/breast-cancer/breast-cancer-new.data", "../data/breast-cancer/breast-cancer.names")
# c1 = C45("../data/cheat/cheat.data", "../data/cheat/cheat.names")
c1.fetchData()
c1.preprocessData()
c1.generateTree()
c1.printTree()
c1.validate()