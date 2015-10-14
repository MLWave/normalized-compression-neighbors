# Normalized Compression Neighbors

NCN(compressor="bzip", nr_estimators=5, sub_sample=1., max_anchor_size=0, random_state=0, verbose=0 )

Parameter | Description
--- | ---
compressor | String. The compressor to use for calculating Normalized Compression Distance. Any of ["bzip","zlib","snappy","lzma"]. * Default: Bzip *
nr_estimators | Int. The number of anchor corpus/neighbors to create. * Default: 5 *
sub_sample | Float. When fitting on a data set, use sub-sampling. `0.7` means 30% of samples are ignored. * Default: 1. *
max_anchor_size | Int. When a random anchor corpus reaches this size (in characters), halt fitting. * Default: 0 (no max size) *
random_state | Int. Seed for replication. * default: 0 *
verbose | Int. Verbosity level. * default: 0 *

Currently only unsupervised (fully random anchors) fitting works.


```
X = ["Document 1", "Document 2", "Document 3"]
y = [1, 0, 1]

clf = NCN()

clf.fit(X, y)

X = clf.transform(X)
```

X is now a vectorized dataset, with for every sample the distances to all the anchors.