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
