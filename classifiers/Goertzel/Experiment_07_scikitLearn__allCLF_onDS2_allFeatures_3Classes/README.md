# Discussions about this experiment

This experiment is made **after fine tuning hyperparameters** on DS1 dataset using random search.

## Experiment goal
Determine which one of the classifier(s) perform better on DS1 dataset of arythimia with Fourier feature extraction method.

## Dataset used:
MIT-BIH__DS1_5classes__Fourier.csv

## Classifiers selected:
- MLP
- Naive-Bayes

All classifiers specs could be seen in the [notebook](./notebook_01-MIT_BIH-5classes.ipynb).


### Brief description:

The fourier feature extraction method returns about 33 features of each hearbeat. No feature selection was performed, so all of them were used as inputs to all classifiers. Some features are not well numeric-represented, so feature selection is **needed** in a close future.

### Results:

To see results look into the [Results folder](./Results/)

No discussions for now.

### Conclusions:

No conclusions, either.


#### Aditional information:

The notebook runs on top of `experiment.py`. [Purge](./Purge/) are files not necessary, most like a separate place for trying stuff.
