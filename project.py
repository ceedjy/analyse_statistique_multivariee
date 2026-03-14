"""
Auteure : Cassiopée Gossin - M1 CMI informatique 
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sn
import pandas as pd

iris = datasets.load_iris()

### Graph visualisation ### 

# display iris 
def visualizeIris():
    _, ax = plt.subplots()
    scatter = ax.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target)
    ax.set(xlabel=iris.feature_names[2], ylabel=iris.feature_names[3])
    _ = ax.legend(
        scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
    )
    plt.title("Iris dataset")
    plt.show()
    
visualizeIris() # to visualize the complete dataset with classes
 
# display a tab 
def visualize(tab, indiceX, indiceY, nomFct): 
    tab = np.array(tab)
    x = tab.T[indiceX] # colonne x 
    y = tab.T[indiceY] # colonne y 
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title("Points " + nomFct)
    plt.show()
    
# display a tab with a cut 
def visualizeSplit(tab, cutting_variable, cutting_value, nomFct):
    tab = np.array(tab)
    x = tab.T[cutting_variable] # colonne x 
    y = 0
    if (cutting_variable == len(tab[0])-1):
        y = tab.T[0] # colonne y
    else :
        y = tab.T[cutting_variable+1] # colonne y 
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.axvline(x=cutting_value,color='red')
    plt.title("Points " + nomFct)
    plt.show()
    
# display iris database and the prediction
def visualizePrediction(observation, prediction):
    _, ax = plt.subplots()
    scatter = ax.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target)
    ax.set(xlabel=iris.feature_names[2], ylabel=iris.feature_names[3])
    pred, = ax.plot(observation[2] ,observation[3], label=prediction, marker="o", color="red")
    first_legend = ax.legend(
        scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
    )
    # we add a second legend to see the predicted class 
    _ = plt.legend(handles=[pred], loc='upper right')
    # Manually add the first legend back to the plot
    ax.add_artist(first_legend)
    # show the graph 
    plt.title("Iris dataset with a prediction")
    plt.show()
    
# display confusion matrix 
def visualizeConfMatrix(matrix):
    df_cm = pd.DataFrame(matrix, index = [i for i in iris.target_names],
                  columns = [i for i in iris.target_names])
    plt.figure(figsize = (6,5))
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion matrix")
    plt.show()
    

### Useful functions ###

# calculate a split 
def split(dataset, target, cutting_variable, cutting_value):
    tab_left = []
    tab_right = []
    target_left = [] # class of the data for the left tab 
    target_right = [] # class of the data for the right tab 
    for i in range(len(dataset)):
        if (dataset[i][cutting_variable] < cutting_value):
            tab_left.append(dataset[i])
            target_left.append(target[i])
        else :
            tab_right.append(dataset[i])
            target_right.append(target[i])
    tab_left = np.array(tab_left)
    tab_right = np.array(tab_right)
    return tab_left, target_left, tab_right, target_right

# calculate the gini of this split 
def gini_of_split(targetset_left, targetset_right):
    Gini_left = calculateGini(targetset_left)
    if (len(targetset_right) != 0): # ici on fait un test car il est possible que les noeud =s aient pas de fils droits mais un fils gauche 
        Gini_right = calculateGini(targetset_right)
    else : 
        Gini_right = 0
    
    totalNbFlower = len(targetset_left) + len(targetset_right)
    left = len(targetset_left) / totalNbFlower
    right = len(targetset_right) / totalNbFlower
    
    Gini_index_of_split = Gini_left * left + Gini_right * right
    return Gini_index_of_split
    
# calculate the gini value for a specific target 
def calculateGini(targetset):
    somme = 0
    for i in range(max(targetset)+1):
        prob = proba(targetset, i)
        somme += prob * (1- prob)
    return somme
    
# calculate the probality of a dataset to have a specific class
def proba(targetset, indiceClass):
    somme = 0
    for i in range(len(targetset)):
        if (targetset[i] == indiceClass):
            somme += 1
    return somme/len(targetset)

# calculate the best split among all variables of the dataset 
def best_split(dataset, targetset):
    smallestGini = 2
    cutting_variable = 0
    cutting_value = 0
    datasetL = 0
    datasetR = 0
    targetsetL = 0
    targetsetR = 0
    for noVar in range(len(dataset[0])): # for each variable of this dataset
        # valueDiv represent the separation of our datas to find the smallest gini and the best cutting 
        # 50 is an arbitrary number -> we separate the datas 50 times 
        valueDiv = (max(dataset[noVar]) - min(dataset[noVar])) / 50
        tempCutting_value = min(dataset[noVar]) 
        for i in range(50):
            dtsetL, target_l, dtsetR, target_r = split(dataset, targetset, noVar, tempCutting_value)
            if (len(target_l) != 0 and len(target_r) != 0):
                gini = gini_of_split(target_l, target_r)
                if (gini < smallestGini):
                    smallestGini = gini
                    cutting_variable = noVar
                    cutting_value = tempCutting_value
                    datasetL = dtsetL
                    datasetR = dtsetR
                    targetsetL = target_l
                    targetsetR = target_r
            tempCutting_value = tempCutting_value + valueDiv
    return smallestGini, cutting_variable, cutting_value, datasetL, targetsetL, datasetR, targetsetR


### Node class ###

class Node:
    # variable initialization 
    def __init__(self, data, target, d):
        self.data = data         # dataset
        self.target = target     # targetset
        self.fd = None           # left son 
        self.fg = None           # right son 
        self.purity = None       # purity of the cut
        self.cutVar = None       # cutting_variable
        self.cutVal = None       # cutting_value 
        self.depth = d           # deph of this node in the tree
        self.nbIndiv = len(data) # number of individual 
        
        
### Useful functions with the tree (Classification Tree object) ###

# recursive function to create a tree 
def make_a_tree(node, data, target):
    if (node.depth > 30 or node.nbIndiv < 5): # if hyper-parameter conditions are met we stop
        purity = gini_of_split(target, [])
        # Updates the attributes/values of the Node
        node.purity = purity 
    else :
        # Splits the data according to the best value and best column
        purity, cutVar, cutVal, dsL, tsL, dsR, tsR = best_split(data, target)
        # Updates the attributes/values of the Node
        node.purity = purity 
        node.cutVar = cutVar
        node.cutVal = cutVal
        # Calls make_a_tree for both divided parts of the region (children of a node) if not pure 
        if (node.purity != 0.0) :
            newNode = Node(dsL, tsL, node.depth+1)
            node.fg = newNode
            node.fd = Node(dsR, tsR, node.depth+1)
            make_a_tree(node.fg, dsL, tsL)
            make_a_tree(node.fd, dsR, tsR)

# create a particular tree 
def fit(data, target):
    # Initialize the root node
    firstNode = Node(data, target, 0)
    # Starts to build the tree (calls the recursive function)
    make_a_tree(firstNode, data, target)
    return firstNode
    
# Check to which terminal node a new observation(s) fells and return its majority class(es)
def predict(newObservation, data, target, printPath):
    tree = fit(data, target) # create the tree
    noClass = travelTree(tree, newObservation, printPath) # travel in the tree to find the good class for this observation 
    print(f'Classe prédite pour {newObservation} est : {iris.target_names[noClass]}\n')
    return iris.target_names[noClass]

# travel in the tree to find the good class for the observation, printpath is here to print the path of an observation if we want to  
def travelTree(tree, obs, printPath):
    if(tree.fg == None and tree.fd == None): # we are on a leaf
        # we are looking for the class that dominate this node 
        noClass = dominantClass(tree.target)
        if (printPath):
            print(f'L observation est arrivée dans la classe {iris.target_names[noClass]}')
        return noClass
    elif (tree.fd == None):
        if (printPath):
            print("Il n'y a pas de fils droit donc je vais dans le fils gauche")
        return travelTree(tree.fg, obs, printPath)
    else : 
        # we search if we need to continue on the left node or the right node 
        if(obs[tree.cutVar] <= tree.cutVal): # if, where the variable of the 2 sons of the node splits, the value of the observation is less than the splitting value 
            if (printPath):
                print(f'L observation a une {iris.feature_names[tree.cutVar]} plus petite que celle du noeud : {obs[tree.cutVar]} <= {tree.cutVal}, donc on va dans le fils gauche')
            return travelTree(tree.fg, obs, printPath) # we continue with the left son
        else :
            if (printPath):
                print(f'L observation a une {iris.feature_names[tree.cutVar]} plus grande que celle du noeud : {obs[tree.cutVar]} > {tree.cutVal}, donc on va dans le fils droit')
            return travelTree(tree.fd, obs, printPath) # we continue with the right son 

# find the dominated class among a targetset
def dominantClass(targetset):
    dico = {} # we use a dictionnary to count numbers of individual of each classes 
    moreRepresented = -1 # the class most represented 
    moreRepValue = -1 # the number of individual of the most represented class
    for elem in targetset: # we put in the dictionnary
        if elem in dico :
            dico[elem] += 1
        else :
            dico[elem] = 1
    for key, value in dico.items(): # we find the most represented class 
        if (value > moreRepValue):
            moreRepValue = value 
            moreRepresented = key
    return moreRepresented

# Count how many times the features were used to build a tree
def feature_importance(tree, features):
    dico = {} # we create a dictionnary to stock teh count of each feature
    for i in range(len(features)):
        dico[i] = 0
    # we travel all the tree to complete the dictionnary of importance
    dico = travelTreeFI(tree, dico)
    # we create a good visualization for the dico
    print("L'importance des features est la suivante :")
    for j in range(len(features)):
        print(f'{features[j]} : {dico[j]}')
    return dico

def travelTreeFI(tree, dico):
    if(tree.fg == None and tree.fd == None):
        return dico
    else : 
        # we need to continue on both sons 
        travelTreeFI(tree.fg, dico) # we continue with the left son
        travelTreeFI(tree.fd, dico) # we continue with the right son
        dico[tree.cutVar] += 1
        return dico


### Train/test split and cross-validation functions ###

# blend the datas
def blend(data, target):    
    data_shuf = []
    target_shuf = []
    index_shuf = list(range(len(data)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        data_shuf.append(data[i])
        target_shuf.append(target[i])
    return data_shuf, target_shuf

# separate data between 2 tabs for train et test (with a certain percentage)
def separateData(data, target, percentTrain, percentTest):
    indice = int(len(data) * percentTrain / 100) # where we need to cut the data 
    DTrain = data[:indice]
    DTest = data[indice:]
    TTrain = target[:indice]
    TTest = target[indice:]
    return DTrain, TTrain, DTest, TTest

# calculate performance (percent error rate) of the result of the test 
def performance(trainData, trainTarget, testData, testTarget):
    tree = fit(trainData, trainTarget) # create the tree with train data
    # instanciate variable to calculate performance 
    numberBadClass = 0
    # the confusion matrix (number of class (observed) x number of class (predicted))
    confMatrix = np.zeros((len(set(testTarget)),len(set(testTarget))))
    for i in range(len(testData)):
        predictedClass = travelTree(tree, testData[i], False) # return the predicted class of this observation 
        # we add the count in the confusion matrix 
        confMatrix[testTarget[i]][predictedClass] += 1
        # test if the observation is good or bad predicted 
        if (predictedClass != testTarget[i]):
            numberBadClass += 1
    percentError =( numberBadClass / len(testData)) *100
    print(f'Le pourcentage d erreur est de {percentError}%')
    print("Le matrice de confusion est la suivante :")
    print(confMatrix)
    visualizeConfMatrix(confMatrix)
    print("")
    return percentError

# cross-validation -> average performance for all setup
def crossValidation(data, target, percentTrain, percentTest):
    # first we blend data to have later a better separation of our data 
    blendedData, blendedTarget = blend(data, target)
    # variable to calculate the average error 
    somme = 0
    # loop to do the cross validation (the range is to loop over all data, even if some of them are taken twice)
    for i in range(int(percentTrain/percentTest)+1): 
        # then we separate the data between the percent of train data and test data
        trainData, trainTarget, testData, testTarget = separateData(blendedData, blendedTarget, percentTrain, percentTest)
        # we calculate the performance of this setup 
        percentError = performance(trainData, trainTarget, testData, testTarget)
        somme += percentError
        # at the end, to change the data and do the cross validation, all we need to do is to put the test data 'at the begining' of the trained data (and the same for the target) and to loop
        blendedData = testData + trainData
        blendedTarget = testTarget + trainTarget
    # we calculate the average of all performance
    averagePerf = somme/(int(percentTrain/percentTest)+1)
    print(f'La moyenne des performance du modèle est la suivante : {averagePerf}%')
    return 


### Tests and calling functions ###

#tab_left, target_left, tab_right, target_right = split(iris.data, iris.target, 2, 2.5)
#print(gini_of_split(target_left, target_right))
#visualize(tab_left, 2, 3, "split left")
#visualize(tab_right, 2, 3, "split right")
#print(iris.data[0][0])

# TP1 : we have found one best split for the current complet dataset (iris)
_, cutting_variable, cutting_value, _, _, _, _ = best_split(iris.data, iris.target)
# we visualize the best split for iris 
visualizeSplit(iris.data, cutting_variable, cutting_value, "with the first split of iris")

# TP2 : we create some predictions based on observations (we use the tree)
fit(iris.data, iris.target) # this create the tree 
# EXAMPLES FOR PREDICTIONS :
print("Exemples de prédictions :")
# we create some observation 
obs1 = [4.6, 3.1, 1.5, 0.2]
obs2 = [6.8, 3.2, 5.9, 2.3]
obs3 = [5.6, 2.5, 3.9, 1.1]
obs4 = [4.6, 3.2, 3.9, 1.8]
obs5 = [4.6, 3.2, 1.9, 1.8]
# we predict
d = iris.data    # data
t = iris.target  # target
prtPath = True   # this variable is used when we want to print the path of an observation in the tree, True = the path is printed, False = the path is not printed
pred1 = predict(obs1, d, t, prtPath)
pred2 = predict(obs2, d, t, prtPath)
pred3 = predict(obs3, d, t, prtPath)
pred4 = predict(obs4, d, t, prtPath)
pred5 = predict(obs5, d, t, prtPath)
# we visualize the data with the prediction 
visualizePrediction(obs1, pred1)
visualizePrediction(obs2, pred2)
visualizePrediction(obs3, pred3)
visualizePrediction(obs4, pred4)
visualizePrediction(obs5, pred5)
# we can also vizualize the importance of each feature in this tree
tree = fit(iris.data, iris.target)
feature_importance(tree, iris.feature_names)
"""
We can notice in the feature importance that the features sepal length and sepal width are not used. 
That's why we choose to visualize the observations only with petal length and petal width axis (features 2 and 3).
"""

# now we do the same but with 70% of the dataset for the model (tree) and 30% for the test
# EXAMPLE FOR PERFORMANCE :
print("\nExemples de performance (70/30 sur iris) :")
# we blend the data
blendedData, blendedTarget = blend(iris.data, iris.target)
# we separate the data
trainData, trainTarget, testData, testTarget = separateData(blendedData, blendedTarget, 70, 30)
# we calculate the performance of the model created by the train data - the metric is the cost of the tree (number of individual in wrong class after the test)
percentError = performance(trainData, trainTarget, testData, testTarget)
# in addition to that, a confusion matrix is created to have a better view over what's happening (in terminal and with a plot)

# now we can put everything together to do the cross validation and calculate the average performance
# EXAMPLE FOR CROSS-VALIDATION :
print("\nExemples de cross-validation (70/30 sur iris) :")
averagePerf = crossValidation(iris.data, iris.target, 70, 30)


