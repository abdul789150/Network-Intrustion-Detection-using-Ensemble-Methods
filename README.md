# Network-Intrustion-Detection-using-Ensemble-Methods

By the help of Ensemble Learning technique, I have created an ML model for cyber attack detections using network traffic dataset.

## UGR’16 Dataset
UGR16 contains network traffic information and is designed to test modern Intrusion Detection Systems (IDS).
UGR’16 is a collection of net flow traces for more than 4 months of traffic in a real network from a tier-3 ISP.

Each month dataset is divided in weeks. The size of dataset is extremely large. For June week 4 uncompressed csv file size is 20GB.
So, we will use a small subset of dataset in which we will have about 500,000 rows of data.

In UGR 16 dataset we have 12 features and a class column. We will be working with 3 classes of attacks in UGR 16 dataset, which are:
- Anomaly-sshscan.
- Anomaly-spam.
- Blacklist.

From 12 features we will be selecting only 5 features. Selected features are:
- Source IP.
- Destination Port.
- Forwarding Status.
- Packets Exchanged.
- Bytes Exchanged

## Classification Model And Results
We will be using AdaBoost classifier using scikit learn library. As per the results from learning curves and model complexity graph, we initialized the model with **6 number of estimators.**
We split the data into training and testing subsets. It took **45 seconds** to train the model, because the number of estimators were small in number.
We get **95.5% accuracy** on testing subset of data.

### Confusion Matrix
![Confusion Matrix Result](https://github.com/abdul789150/Network-Intrustion-Detection-using-Ensemble-Methods/blob/main/cm.png)


## Libraries Required
- numpy
- pandas
- scikit-learn
- matplotlib

## How to run this project
For running this project you can run the main_file.py by using the command
```
python main_file.py
```

For more information about the project check **Model Notebook.ipynb** file.
