##  Test of dense net at end of CNN Encoder

**Github humpback call database is still too large to upload.  I am reducing size.  Use the fake data option**

# My Questions are:

Why does the NN not converge and detect the labelled calls?
Why does the accuracy of the training set increase nicely but
     when the labels of the training set are predicted, they are not at all accruate?
      
Here is the confusion matrix after 10 epochs:

Confusion matrix fractions for predictions on dataset train dataset of length 828


|        PREDICT |     0     |      1     |
| -------------- |:---------:|:----------:|
|   Label = 0    |  TN 0.390 |   FN 0.136 |
|   Label = 1    |  FP 0.278 |   TP 0.196 |
 
      
