# CDSC_AL: A Clustering-based Data Stream Classification framework using Active Learning

The "Supplemental Result.pdf" includes the results for comparison with semi-supervised methods using 5%, 15%, 20% labeled data. Also, the comparison results between supervised methods and CDSC-AL method with 5%, 15%, and 20% labeled data respectively.

There are two codes with different settings for the benchmark data streams:

  1. The main_final_draft.py file is developed for arranging data streams to have abrupt drifts and run this code on 
  
    Synthetic-1, Synthetic-2, Sea, and Shuttle
  
  2. The main_final_draft4.py file is developed for simulating data streams with gradual concept drift and run this code on 
  
    KDD cup 99, Forest covtype, Gas Sensor Drift, MNIST, CiFAR-10

The two synthetic datasets (Synthetic-1 and Synthetic-2) are generated by the authors and thus we include them here. For the remaining seven datasets, it can found at following links:
  1. https://archive.ics.uci.edu/ml/index.php
  
  2. http://users.rowan.edu/ ∼polikar/nse.html
  
To run these two files with different datasets, go to line 17 to change the name of dataset

In line 11, the global variable label_ratio allows for users to change the proportion of labeled data in each incoming data chunk

Two different evaluation metrics are used: 
  
  1. BAcc1Hist: A vector of the Balanced Classification Accuracy values for the entire data streams
  
  2. F1Hist: A vector of the Macro-average values of the F1-score for the entire data streams
