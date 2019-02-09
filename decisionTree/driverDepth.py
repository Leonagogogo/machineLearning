from DecisionTree import *
import pandas as pd
from sklearn import model_selection


header = ["Prengant times", "Plasma glucose", "Diastolic blood pressure", "Triceps skin fold thickness", "2-Hour serum insulin", "Body mass index", "Diabetes pedigree function", "Age", "Class"]
df = pd.read_csv('https://raw.githubusercontent.com/anushamanur/Pima/master/pima_indians_diabetes.csv', header=None, names=header)

lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()
t_train = build_tree(train, header)

depth = getDepth(t_train)
# print("*************Tree before pruning*******")
# print_tree(t)
acc1 = computeAccuracy(test, t_train)
print("Accuracy on test (before pruning)= " + str(acc1))

## TODO: You have to decide on a pruning strategy
# print("*************After Pruning Depth************")
t_train_depth = t_train
for nodedepth in depth:
	# print(nodedepth)
	# print(acc1)
	t_pruned = prune_tree_depth(t_train_depth, getMaxDepth(t_train_depth)-nodedepth)	
	# print_tree(t_pruned)
	acc4 = computeAccuracy(test, t_pruned)
	# print("Accuracy on test = " + str(acc4))

	if acc4 > acc1:		
		t_train_depth = t_train
		print("Pruning Node Depth Accuracy Promote = acc1: ", acc1, "acc4: ", acc4)
		print_tree(t_pruned)
		acc4 = 0



		





