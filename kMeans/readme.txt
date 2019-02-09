Language: python2

Build and Compile:
python tweets-k-means.py numberOfcluster initialSeedsFile TweetsDataFile outputFile
e.g.
python tweets-k-means.py 15 InitialSeeds.txt Tweets.json tweets-k-means-output.txt


attention:
   the number of cluster must be consistent with the initial seeds number
   or the code could not be executed

output file:
   tweets-k-means-output.txt
   Each line represents a cluster. The format is: cluster: list of tweets ids belongs to this cluster
   The last line is SSE

Library used:
   sklearn.feature_extraction.text CountVectorizer

