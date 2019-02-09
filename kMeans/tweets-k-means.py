
import json
from pprint import pprint
from sklearn.metrics import jaccard_similarity_score
from sklearn.feature_extraction.text import CountVectorizer
import sys

#read JSON file 
#retrieve "id" and "text" information
def readTweetsDataFile():
    oriData = {}
    with open(sys.argv[3]) as f:
        for line in f:
            fline = json.loads(line)
            key = fline["id"]
            value = fline["text"]
            oriData[key] = value
    return oriData

#read initial seeds file
def readInitialSeedsFile():
    initialSeeds=[]
    with open(sys.argv[2]) as f:
        for line in f:
            fline = line.strip('\n')
            initialSeeds.append(int(fline.replace(',','')))
    return initialSeeds

#calculate the distance between two string 
#using Jaccad method
def jaccard_similary(str1, str2):
    a = set(str1)
    b = set(str2)
    inter = a.intersection(b)    
    union = a.union(b)     
    return float(1.0 - float(len(inter))/float(len(union)))

#tokenizer text values
def calculate_tokenizer(dict, x_id, y_id):
    vectorizer_X = CountVectorizer()
    X = vectorizer_X.fit_transform([dict[x_id]])
    vectorizer_Y = CountVectorizer()
    Y = vectorizer_Y.fit_transform([dict[y_id]])    
    similar = float(jaccard_similary(vectorizer_X.get_feature_names(), vectorizer_Y.get_feature_names()))
    return similar

#according to the known seeds to cluster
def calculate_cluster(centroid, dict):
    result = {}
    for i in centroid:    
        values = []
        result[i] = values
        for j in dict.keys():           
            similarity = calculate_tokenizer(dict, i, j)            
            if similarity < 0.3:
                result[i].append(j)
    return result


#according to the known cluster to get the new centroid
def update_centroid(dict1,dict2):
    update=[]    
    for cen in dict2.keys():
        allDist = 0
        result = 1000
        newCen = cen
        values = []
        values = dict2[cen]
        for x in values:           
            for y in values:               
                allDist = allDist+calculate_tokenizer(dict1, x, y)            
            if allDist < result:
                result = allDist             
                newCen = x
            allDist = 0
        update.append(newCen)
        
    return update 

#calculate the similarity between previous centroid and current centroid
def stop_recog(oldCen, newCen):
    old = set(oldCen)
    new = set(newCen)
    same = new.intersection(old)
    if len(same) == len(oldCen):
        return True
    return False

##the previous centroid is same with the newest centroid
##the update stops
def stop(oldCentroid, newCentroid, oriData):
    count = 1000
    while stop_recog(oldCentroid, newCentroid)==False:
        count = count-1
        oldCentroid = newCentroid
        current = calculate_cluster(oldCentroid, oriData)
        newCentroid = update_centroid(oriData, current)
        if count==0:
            return False
    return True

##calculate the SSE and save the result into file
def calculate_SSE(result, dict):   
    distance = 0
    file = open(sys.argv[4],"w")
    for key in result.keys():
        values = result[key]
        file.writelines(str(key)+str(":")+str(values)+"\n")     
        for value in values:         
            distance = calculate_tokenizer(dict, key, value)**2 + distance
    file.writelines("SSE is: "+str(distance)+"\n")    
    file.close()       

if __name__ == "__main__":
    #read tweets data 
    TweetsData={}
    TweetsData = readTweetsDataFile()
    #read initial seeds file
    initialSeeds=[]
    initialSeeds = readInitialSeedsFile()

    k = int(sys.argv[1])
    if k != len(initialSeeds):
        print("Cluster number should be consistent with initial seeds number!")
        sys.exit()
    #the first time to cluster
    #accroding to the initial centroid
    currentCentroid = {}
    currentCentroid = calculate_cluster(initialSeeds, TweetsData)
    #update centroid 
    newCentroid = update_centroid(TweetsData, currentCentroid)
    oldCentroid = initialSeeds
    #judge whether the update is stopped
    if stop(oldCentroid, newCentroid, TweetsData) == True:
        print("Success to Update")
        #update the newest cluster
        SSE={}
        SSE = calculate_cluster(newCentroid, TweetsData)
        #save file
        calculate_SSE(SSE, TweetsData)
    else:
        print("Failure to Update")










