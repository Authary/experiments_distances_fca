import pandas as pd
import PCA
import FCAD
import random


def dfToContext(df) :
    dict1 = {}
    compteurAttr = 0
    for col in df :
        dict1[col] = dict()
        for attr in df[col].unique().tolist():
            dict1[col][attr] = compteurAttr
            compteurAttr += 1

    incidence = list()
    compteurObj = 0
    for row in df.index :
        for col in df :
            incidence.append([compteurObj, dict1[col][df.at[row,col]]])
        compteurObj += 1
    return [incidence, compteurObj, compteurAttr]

def downSampleContext(context, fracRow, fracCol) :
    
    sampledRows = random.sample(range(context[1]), int(fracRow * context[1]))
    sampledColumns = random.sample(range(context[2]), int(fracCol * context[2]))
    newIncidence = list()
    indexRow = 0
    for row in sampledRows :
        indexCol = 0
        for col in sampledColumns :
            if [row, col] in context[0] :
                newIncidence.append([indexRow, indexCol])
            indexCol += 1
        indexRow += 1
    return [newIncidence, len(sampledRows), len(sampledColumns)]
        

"""
Expes sur le dataset Mushroom
"""
nb_iter = 1500
filepath = "agaricus-lepiota.data"
df = pd.read_csv(filepath, header = None)
print("created the df")

context = dfToContext(df)
with open("densitesMushroom.txt", "a") as my_file:
    my_file.write(str(len(context[0]) / (context[1] * context[2])) + "\n")



for i in range(nb_iter) :
    context1 = downSampleContext(dfToContext(df), 0.007, 0.12)
    context2 = downSampleContext(dfToContext(df), 0.007, 0.12)
    while context1[2] != context2[2] :
        context1 = downSampleContext(dfToContext(df), 0.007, 0.12)
        context2 = downSampleContext(dfToContext(df), 0.007, 0.12)
    with open("contextsMushroom.txt", "a") as my_file:
        my_file.write(str(context1[1]) + " " + str(context1[2]) + "\n")
    with open("densitesMushroom.txt", "a") as my_file:
        my_file.write(str(len(context1[0]) / (context1[1] * context1[2])) + " " + str(len(context2[0]) / (context2[1] * context2[2])) + "\n")

    factual_distance = FCAD.factual_distance(context1, context2, 2)

    conceptual_distance = FCAD.conceptual_distance(PCA.concepts(context1), PCA.concepts(context2), 2, 1)
    
    proper1, _, _ = PCA.properPremises(context1)
    proper2, _, _ = PCA.properPremises(context2)
    
    logical_distance = FCAD.logical_distance(proper1, proper2, context1[2], 2, 1)
    with open("resultatsMushroom56x14.txt", "a") as my_file:
        my_file.write(str(factual_distance) + " " + str(conceptual_distance) + " " + str(logical_distance)  + "\n")

"""
Expes sur le dataset Bob Ross
"""            

nb_iter = 1500
filepath = "elements-by-episode.csv"
df = pd.read_csv(filepath)
context = dfToContext(df)

with open("densitesBobRoss.txt", "a") as my_file:
    my_file.write(str(len(context[0]) / (context[1] * context[2])) + "\n")


df = df.drop(labels = "TITLE", axis = 1)
df = df.drop(labels = "EPISODE", axis = 1)
for i in range(nb_iter) :
    context1 = downSampleContext(dfToContext(df), 0.12, 0.20)
    context2 = downSampleContext(dfToContext(df), 0.12, 0.20)
    while context1[2] != context2[2] :
        context1 = downSampleContext(dfToContext(df), 0.12, 0.20)
        context2 = downSampleContext(dfToContext(df), 0.12, 0.20)
    with open("contextsBobRoss.txt", "a") as my_file:
        my_file.write(str(context1[1]) + " " + str(context1[2]) + "\n")
    with open("densitesBobRoss.txt", "a") as my_file:
        my_file.write(str(len(context1[0]) / (context1[1] * context1[2])) + " " + str(len(context2[0]) / (context2[1] * context2[2])) + "\n")

    factual_distance = FCAD.factual_distance(context1, context2, 2)

    conceptual_distance = FCAD.conceptual_distance(PCA.concepts(context1), PCA.concepts(context2), 2, 1)
    
    proper1, _, _ = PCA.properPremises(context1)
    proper2, _, _ = PCA.properPremises(context2)
    
    logical_distance = FCAD.logical_distance(proper1, proper2, context1[2], 2, 1)
    with open("resultatsBobRoss.txt", "a") as my_file:
        my_file.write(str(factual_distance) + " " + str(conceptual_distance) + " " + str(logical_distance)  + "\n")
            