# Run neural net with fake data

if useFakeData:
    h5filename = buildFakeSpectra(1000, 256, 256)
    
    generates (1000, 256, 256) fake calls and backgrounds
    
![Fake call](https://github.com/veirs/orcasound-signal-classifier/blob/main/notes/fakeCall.png "Fake Call")
![Fake background](https://github.com/veirs/orcasound-signal-classifier/blob/main/notes/fakeBackground.png "Fake Background")   

A run of 10 epochs give these loss and accruacy plots:

![Loss](https://github.com/veirs/orcasound-signal-classifier/blob/main/notes/fakeModelLoss.png "Loss history")
![Accuracy](https://github.com/veirs/orcasound-signal-classifier/blob/main/notes/fakeModelAccuracy.png "Accruacy history")   

And these confusion matrices (for one batchsize of records):

Confusion matrix fractions for predictions on dataset **train** dataset of length 100


|        PREDICT |     0     |      1     |
| -------------- |:---------:|:----------:|
|   Label = 0    |  TN 0.550 |   FN 0.010 |
|   Label = 1    |  FP 0.000 |   TP 0.440 |

Confusion matrix fractions for predictions on dataset **test** dataset of length 100


|        PREDICT |     0     |      1     |
| -------------- |:---------:|:----------:|
|   Label = 0    |  TN 0.170 |   FN 0.330 |
|   Label = 1    |  FP 0.100 |   TP 0.400 |


**If the failure to classify the test dataset is due to 'over fitting', what should be done? **

# Run with larger fake dataset?

## Try run 40 epochs on fake dataset of 7000 records

Save model as: models/Classify_h5fakeSpecsSml_[256-128-32-8]_Em_h5_0_40_epochs/

Confusion matrix fractions for predictions on dataset **train** dataset of length 100


|        PREDICT |     0     |      1     |
| -------------- |:---------:|:----------:|
|   Label = 0    |  TN 0.330 |   FN 0.190 |
|   Label = 1    |  FP 0.000 |   TP 0.480 |

Confusion matrix fractions for predictions on dataset **test** dataset of length 100


|        PREDICT |     0     |      1     |
| -------------- |:---------:|:----------:|
|   Label = 0    |  TN 0.360 |   FN 0.120 |
|   Label = 1    |  FP 0.100 |   TP 0.520 |

**Weirdly**, now the NN does better on the **test** dataset

![Loss - Accuracy](https://github.com/veirs/orcasound-signal-classifier/blob/main/notes/7000_40_FakeRecords.png "Loss&Accruacy history")  

What happened around epoch 12 in the Loss and then around epoch 40 in the Accuracy?

## Try Dropout somewhere? 

