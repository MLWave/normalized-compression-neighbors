"""
  Normalized Compressed Neighbours
  
  Online/out-of-memory version of Normalized Compression Distance [1].
  
  We create n small corpora from random documents in train set. We call this an anchor corpus.
  
  For every sample in train and test set, we calculate the NCD between the sample and all the anchor corpora.
  
  Allows for both supervised and unsupervised neighbours. In the supervised case, documents with the same class are 
  added to the same anchor corpus. In the unsupervised case, documents are selected randomly.
  
  In the zero-shot case, the anchor corpora are filled with documents from Wikipedia, IMDB, Newsgroup20, Gutenberg and Stackoverflow.
  
  The resulting vectors are the normalized distances between a sample and all the anchor corpora. Such a vector can be 
  seen as an entry in a semantic knowledge base [2]. For instance, when a anchor corpora contains documents with the theme "terrorism"
  then a document with a similar theme, will get a closer distance to such an anchor corpora, than a non-related document. 
  With distinct and carefully crafted anchor corpora, the vectors can hold information on: "language used, sentiment, category, 
  word syntax/style" and more.
  
  The resulting vectors can be used both supervised and unsupervised again: In the supervised case we train an SVM with a 
  linear kernel on the compression vectors. In the unsupervised case we use Mapper [3] or plain clustering algorithms to find 
  similar documents.
  
  Currently supports compressors: Bzip, Zlib, PyLZMA, Snappy. Tested on Python 2.7.
  
  [1] http://homepages.cwi.nl/~paulv/papers/cluster.pdf
  [2] http://www.cs.cmu.edu/~./fmro/papers/zero-shot-learning.pdf
  [3] https://research.math.osu.edu/tgda/mapperPBG.pdf
  
"""
import random
import sys
from math import log

class NCN(object):
  def __init__(self, compressor="bzip", nr_estimators=5, sub_sample=1., max_anchor_size=0, random_state=0, verbose=0 ):
    if compressor == "bzip":
      from bz2 import compress
    if compressor == "snappy":
      from snappy import compress
    if compressor == "lzma":
      from pylzma import compress
    if compressor == "zlib":
      from zlib import compress
    self.compressor = compressor
    self.compress = compress
    self.nr_estimators = nr_estimators
    self.sub_sample = sub_sample
    self.random_state = random_state
    self.max_anchor_size = max_anchor_size
    self.verbose = verbose
    
  def __repr__(self):
    return str("""NCN(compressor="%s", nr_estimators=%s, random_state=%s)"""%(self.compressor, self.nr_estimators, self.random_state))

  def ncd(self,anchor,x):
    C_x = len(self.compress(x.encode('utf8')))
    C_y = anchor[1]
    C_x_y = len(self.compress(x.encode('utf8') + " " + anchor[0]))
    return (C_x_y - min(C_x,C_y)) / float(max(C_x,C_y)) 
    
  def fit(self, X, y=[]):
    if self.verbose > 0:
      print("""Creating %s anchors """%self.nr_estimators)
    random.seed(self.random_state)
    anchors = []
    for i in range(self.nr_estimators):
      anchors.append(["".encode('utf8'),0])
    if len(y) == 0: # Unsupervised, so pick fully random sample 
      for x in X:
        if random.random() < self.sub_sample:
          random_anchor_id = random.randint(0,self.nr_estimators-1)
          anchors[random_anchor_id][0] += " " + x.encode('utf8') # add the data to a random anchor
          if self.max_anchor_size > 0 and len(anchors[random_anchor_id][0]) > self.max_anchor_size:
            break # halting because max characters in an anchor was found
            
      # Pre-Calculating compressed lengths
      for anchor in anchors:
        anchor[1] = len(self.compress(anchor[0]))
    self.anchors = anchors
    if self.verbose > 0:
      print("""Done creating anchors. Result [(len, compressed_len)]: %s """%
        (sorted([(len(f[0]),f[1]) for f in self.anchors],reverse=True)))
  
  def transform(self, X, type="list"):
    try:
      self.anchors = self.anchors
    except:
      sys.exit("""You need to fit first (create anchors) before you can calculate an anchor distance.""")
    if self.verbose > 0:
      print("""Transforming Data. Type: %s"""%(type))
    X_anchor_distances = []  
    for x in X:
      X_anchor_distance = []
      for anchor in self.anchors:
        X_anchor_distance.append( self.ncd(anchor,x) )
      X_anchor_distances.append(X_anchor_distance)
    return X_anchor_distances
    
  def transform_iter(self, X, type="list"):
    try:
      self.anchors = self.anchors
    except:
      sys.exit("""You need to fit first (create anchors) before you can calculate an anchor distance.""")
    if self.verbose > 0:
      print("""Transforming Data. Type: %s"""%(type))
    for x in X:
      X_anchor_distance = []
      for anchor in self.anchors:
        X_anchor_distance.append( self.ncd(anchor,x) )
      yield X_anchor_distance