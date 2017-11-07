from DecisionTree import Decision_Tree
from pprint import pprint
tree = Decision_Tree()
data, labels = tree.example_dataset()
myTree = tree.build_Tree(data, labels)
pprint(myTree)
