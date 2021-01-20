# CDSC_AL: A Clustering-based Data Stream Classification framework using Active Learning

Run the main_final_draft4.py file 

To run this code with different datasets, go to line 17 to change the name of dataset

In line 11, the global variable label_ratio allows for users to change the proportion of labeled data in each incoming data chunk

Two different evaluation metrics are used: 
  
  BAcc1Hist: A vector of the balanced classification accuracy values for the entire data streams
  
  F1Hist: A vector of the macro-average values of the F1-score for the entire data streams
