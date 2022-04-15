# Run neural net with fake data

if useFakeData:
    h5filename = buildFakeSpectra(1000, 256, 256)
    
    generates (1000, 256, 256) fake calls and backgrounds
    
![alt text](https://github.com/veirs/orcasound-signal-classifier/notes/fakeCall.png "Fake Call")
![alt text](https://github.com/veirs/orcasound-signal-classifier/notes/fakeBackground.png "Fake Background")   

A run of 10 epochs give these loss and accruacy plots:

![alt text](https://github.com/veirs/orcasound-signal-classifier/notesfakeModelLoss.png "Loss history")
![alt text](https://github.com/veirs/orcasound-signal-classifier/notes/fakeModelAccuracy.png "Accruacy history")   

And these confusion matrices (for one batchsize of records:

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




