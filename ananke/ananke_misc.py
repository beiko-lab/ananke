import numpy as np
from scipy.sparse import csr_matrix

def translate_otus(input_fasta, input_clusters, output_clusters):
  #  Purpose: take in the input FASTA file, and translate
  #  them to their sequence hashes
  #  Assumption: if two sequences are bit-for-bit identical,
  #  the clustering scheme put them in the same cluster
  hashSet = set()
  labelSet = set()
  label2hash = {}
  print("Reading in sequences...")
  with open(input_fasta) as infasta:
    label = infasta.readline()
    while label:
      label = label[1:].strip()
      seqhash = hash(infasta.readline().strip())
      if seqhash not in hashSet:
        label2hash[label] = seqhash
        hashSet.add(seqhash)
        labelSet.add(label)
      label = infasta.readline()
  print("Translating clusters...")
  outfile = open(output_clusters,"w")
  with open(input_clusters) as inotus:
    for line in inotus:
      line = line.strip().split("\t")
      otu = line[0]
      labels = line[1:]
      outfile.write(otu)
      for label in labels:
        if label in labelSet:
          outfile.write("\t"+str(label2hash[label]))
      outfile.write("\n")

def rarefy_even(timeseriesdb):
  matrix = timeseriesdb.get_sparse_matrix()
  matrix = matrix.todense()
  subsampled_matrix = np.zeros_like(matrix)
  print("Original shape: " + str(matrix.shape))
  #  Get the lowest sample depth
  min_depth = min(np.ravel(matrix.sum(0)))
  print("Minimum sequence depth: " + str(min_depth))
  if (min_depth == 0):
    raise ValueError("Error: Lowest sample depth equal to zero.")
  for col_index in range(matrix.shape[1]):
    nsample = min_depth
    counts = np.ravel(matrix[:,col_index])
    indices = np.arange(len(counts))
    subsampled_counts = np.zeros(len(counts))
    while (nsample > 0):
      indices = indices[counts > 0]
      counts = counts[counts > 0]
      count = np.random.randint(len(counts))
      counts[count] -= 1
      subsampled_counts[indices[count]] += 1
      nsample -= 1
    subsampled_matrix[:,col_index] = np.transpose(np.matrix(subsampled_counts))
  del matrix
  nonzero_rows = np.where(subsampled_matrix.sum(1) > 0)[0]
  #  Get rid of zero-count sequences/rows
  subsampled_matrix = subsampled_matrix[np.ravel(subsampled_matrix.sum(1) > 0),:]
  nobs = np.count_nonzero(subsampled_matrix)
  ngenes = subsampled_matrix.shape[0]
  #  Fetch the existing gene arrays so we don't lose data
  #  TODO: filter "genes/clusters" array
  #  for now, always do clustering after rarefaction
  for gene_array in ["genes/sequences", "genes/sequenceids", \
                     "genes/sequenceclusters", "genes/taxonomy"]:
    arr = timeseriesdb.get_array_by_chunks(gene_array)
    arr = arr[nonzero_rows]
    timeseriesdb.insert_array_by_chunks(gene_array, arr)
    del arr
  #  Shrink file down to the correct size
  timeseriesdb.resize_data(ngenes = ngenes, nobs = nobs)
  sparse_matrix = csr_matrix(subsampled_matrix)
  print("Filtered shape: " + str(sparse_matrix.shape))
  #  Overwrite the old matrix with subsampled one
  timeseriesdb.insert_array_by_chunks("timeseries/data", sparse_matrix.data)
  timeseriesdb.insert_array_by_chunks("timeseries/indptr", sparse_matrix.indptr)
  timeseriesdb.insert_array_by_chunks("timeseries/indices", sparse_matrix.indices)
  print("Rarefaction complete!")
