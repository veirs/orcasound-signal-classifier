##  Test of dense net at end of CNN Encoder

**Github humpback call database is still too large to upload.  I am reducing size.  Use the fake data option**

# My Questions are:

Why does the NN not converge and detect the labelled calls?
Why does the accuracy of the training set increase nicely but
     when the labels of the training set are predicted, they are not at all accruate?
      
Here is the confusion matrix after 10 epochs:

Confusion matrix fractions for predictions on dataset train dataset of **fake data** of length 828


|        PREDICT |     0     |      1     |
| -------------- |:---------:|:----------:|
|   Label = 0    |  TN 0.390 |   FN 0.136 |
|   Label = 1    |  FP 0.278 |   TP 0.196 |
 
Starting with NN with dropout trained on 7000 fake samples and then training on the ~10,000 reccords (~1/2 background, ~1/2 labeled calls, with dropout added after the first Dense layer, gives:

![Loss - Accuracy](https://github.com/veirs/orcasound-signal-classifier/blob/main/notes/HW_all_Dropout_27_Epochs.png "Loss&Accruacy history")    

Confusion matrix fractions for predictions on dataset train dataset of length 6648


|        PREDICT |     0     |      1     |
| -------------- |:---------:|:----------:|
|   Label = 0    |  TN 0.317 |   FN 0.184 |
|   Label = 1    |  FP 0.175 |   TP 0.332 |

Confusion matrix fractions for predictions on dataset test dataset of length 1758


|        PREDICT |     0     |      1     |
| -------------- |:---------:|:----------:|
|   Label = 0    |  TN 0.311 |   FN 0.191 |
|   Label = 1    |  FP 0.166 |   TP 0.332 |    

      
