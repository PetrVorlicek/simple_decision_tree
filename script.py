# Import pandy a dat z csv
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
df = pd.read_csv("./drug200.csv")

# Úprava dat pro učení - strom neumí pracovat se spojitými veličinami
drug_data = df.loc[:, df.columns != "Drug"]
drug_data["Age"] = pd.cut(df.loc[:,"Age"], bins=4, labels=False, right=True)
drug_data["Na_to_K"] = pd.cut(df.loc[:,"Na_to_K"],
 bins=6, labels=False, right=True)
drug_data = drug_data.to_numpy()

# Značky dat
drug_labels = df["Drug"].to_numpy()

###Rozhodovací strom###
from collections import Counter

#Gini impurity: 1 - (procento labelu ve skupině)^2
#labels = vektor značek
def gini(labels):
    impurity = 1
    #počet výskytů labelů
    uniqueCounts = Counter(labels)
    
    #odečítání (procent labelu ve skupině)^2 pro každý label
    for label in uniqueCounts:
        label_probability = uniqueCounts[label] / len(labels)
        impurity -= label_probability ** 2
    return impurity

# Weighted Information gain: gini původních labels - gini rozdělených labels
# Umožňuje porovnávat rozdělení na různých vlastnostech
# unsplit = vektor značek před rozdělením
# splits = vektor vektorů značek po rozdělení na každé možné vlastnosti
def info_gain(unsplit, splits):
    #Gini původního labelu
    gain = gini(unsplit)
    #Výpočet váženého information gain
    for split in splits:
        gain -= gini(split) * len(split)/len(unsplit)
    return gain

# Tato metoda vrací unikátní možné hodnoty, kterých nabývají buňky dat
def get_unique (data, column):
    unique_list = []
    for i in range(len(data)):
        if data[i][column] not in unique_list:
             unique_list.append(data[i][column])
    return unique_list

# Rozdělení dat dle hodnot ve sloupci
def split(data, labels, column):
    data_subsets = []
    label_subsets = []
    
    features = get_unique(data, column)
    
    for feature in features:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(data)):
            if data[i][column] == feature:
                new_data_subset.append(data[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)

    return data_subsets, label_subsets

# Výpočet nejlepšího weighted information gain pro všechny možné rozdělení (splity) dat
def best_split(data, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(data[0])):
        data_subsets, label_subsets = split(data, labels, feature)
        gain = info_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain

# Třídy Internal_node a Leaf jsou třídy, které slouží k vybudování rozhodovacího stromu. 
# Třída Leaf je list stromu, data zde již jsou rozdělená.
# Třída Internal_node reprezentuje větvení rozhodovacího stromu
class Leaf:
    def __init__(self, labels, value):
        self.labels = Counter(labels)
        self.value = value
        
class Internal_node:
    def __init__(self, feature, branches, value):
        self.feature = feature
        self.branches = branches
        self.value = value

# Budování stromu je rekurzivní funkce.
# Funkce hledá nejlepší split pro data, která dostane.
# Pokud není dosaženo žádného information gain, nelze smysluplně data dále dělit 
# a je vrácen Leaf.
# Pokud jsou data rozdělena s nenulovým information gainem, je pro všechny
# podmnožiny opět volána funkce build_tree, a je vrácen Internal_node.
def build_tree(data, labels, value =""):
    best_feature, best_gain = best_split(data, labels)
    if(best_gain == 0):
        return Leaf(Counter(labels), value)
    data_subsets, label_subsets = split(data, labels, best_feature)
    branches = []
    for i in range(len(data_subsets)):
        branch = build_tree(data_subsets[i], label_subsets[i],
         data_subsets[i][0][best_feature])
        branches.append(branch)
    return Internal_node(best_feature, branches, value)

# Klasifikace pomocí stromu prochází strom dle zadaného datapointu,
# dokud nedojde na Leaf, kdy vrací nejčastější hodnotu z daného Leaf
def classify(point, tree):
    if type(tree) is Leaf:
        return tree.labels.most_common(1)[0][0]
    value = point[tree.feature]
    for branch in tree.branches:
        if branch.value == value:
            return classify(point, branch)

# Primitivní funkce na zobrazení struktury stromu.
def read_tree(tree):
    if type(tree) is Leaf:
        print(tree.labels)
        return
    print(tree.feature)
    for branch in tree.branches:       
        read_tree(branch)
    return

###Použití stromu###

my_tree = build_tree(drug_data, drug_labels)

# Hodnocení stromu: správně určeno/počet značek
correct = 0
for i in range(len(drug_data)):
    if classify(drug_data[i], my_tree) == drug_labels[i]:
        correct += 1     
print(correct / len(drug_labels))

# Klasifikace bodu
point = classify([3,"M","HIGH","NORMAL",2],my_tree)
print(point)