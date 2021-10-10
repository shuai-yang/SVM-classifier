# SVM-classifier
we will use Scikit-learn, a Python framework for machine learning, for creating our SVM classifier.
 step-by-step example of how to generate a sample dataset, build the SVM classifier, train it, and visualize the decision boundary that has emerged after training
 
  classifying new samples according to a preexisting set of decision criteria.
  
  binary classification scenario.
We’re going to build a SVM classifier step-by-step with Python and Scikit-learn. This part consists of a few steps:

Generating a dataset: if we want to classify, we need something to classify. For this reason, we will generate a linearly separable dataset having 2 features with Scikit’s make_blobs.
Building the SVM classifier: we’re going to explore the concept of a kernel, followed by constructing the SVM classifier with Scikit-learn.
Using the SVM to predict new data samples: once the SVM is trained, it should be able to correctly predict new samples. We’re going to demonstrate how you can evaluate your binary SVM classifier.
Finding the support vectors of your trained SVM: as we know, support vectors determine the decision boundary. But given your training data, which vectors were used as a support vector? We can find out – and we will show you.
Visualizing the decision boundary: by means of a cool extension called Mlxtend, we can visualize the decision boundary of our model. We’re going to show you how to do this with your binary SVM classifier.
