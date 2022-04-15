##  Test of dense net at end of CNN Encoder

# My Questions are:

why does the NN not converge and detect the labelled calls?
why does the accuracy of the training set increase nicely but
     when the labels of the training set are predicted, they are not at all accruate?
      
here is the confusion matrix after 10 epochs:

Confusion matrix fractions for predictions on dataset train dataset of length 828
             PREDICT   0            1
   Label = 0      TN 0.390    FN 0.136
   Label = 1      FP 0.278    TP 0.196

      
