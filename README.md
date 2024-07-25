# Parametric t-SNE  
----------
This research was conducted by the students André de Macedo Wlodkowski @andrewlod and Kalebe Rodrigues Szlachta @kalebers, mentored by professor Jean Paul Barddal @jpbarddal at Pontifical Catholic University of Paraná
for the Computer Science graduation final project. 

ENGLISH: Archives referring to the prototype are inside the folder parametric-tsne-keras. To execute, it's necessary to follow the software requirements presented on chapter 4 of the artefact.

Main archives: 
- TSNEClassifier.ipynb: file that contains all experiments reffering to the research and implementation of the TSNEClassifier class. Executed on Anaconda environment with Jupyter lab commands.
  
- parametric_tsne.py: main file for Parametric t-SNE implementation, by Luke Lee.

----------

t-distributed stochastic neighbor embedding, abbreviated as t-SNE, provides the novel method to apply non-linear dimensionality reduction technique that preserves the local structure of original dataset. However, in order to transform newly prepared points, a model must be re-trained with whole dataset. This would be extremely inefficient provided that our previous dataset describes the plausible distribution already. Parametric t-SNE instead gives you an explicit mapping between original data and the embedded points. It is achieved by building a parametric model for prediction and training it using the same loss as t-SNE.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

This program was tested under Python 3.6. All necessary packages are contained inside `requirements.txt`.

### Installing

After cloning this repository, install required packages by running the following:

```
pip3 install -r requirements.txt
```

## Running the Tests

`parametric_tsne.py` can be run directly from command-line. See help for details.

```
python3 parametric_tsne.py -h
```

## Deployment

Simply create `ParametricTSNE` instance. The interface was designed similarly to that of scikit-learn estimators.

```python
from parametric_tsne import ParametricTSNE

transformer = ParametricTSNE()

# suppose you have the dataset X
X_new = transformer.fit_transform(X)

# transform new dataset X2 with pre-trained model
X2_new = transformer.transform(X2)
```

## Built With

- [scikit-learn](http://scikit-learn.org/stable/) - Extensive machine learning framework

- [Keras](https://keras.io) - Deep learning framework wrapper that supports TensorFlow, Theano, and CNTK

## Authors

- __Luke Lee__ - Research and implementation - [luke0201](https://github.com/luke0201)

## Acknowledgements

- This project was forked from [zaburo-ch's implementation](https://github.com/zaburo-ch/Parametric-t-SNE-in-Keras).
