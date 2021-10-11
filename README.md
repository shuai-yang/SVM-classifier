# SVM-classifier
This is step-by-step a practice to create a simple SVM classifier for a binary classification scenario using Scikit-learn.
## Steps
### (1) Generating a dataset
![](images/Figure_1.png)<br/>
### (2) Building the SVM classifier
- Choosing a kernel function <br/>
- Fitting training data to the classifier <br/>
### (3) Using SVM to predict new data samples
- Predicting new data samples
- Evaluate the classifier with the test set. In this case, I did so by means of a **confusion matrix**, which shows us the correct and wrong predictions in terms of **true positives, true negatives, false positives and false negatives**<br/>
![](images/Figure_2.JPG)<br/>
In my case, I have 99% true positives and 100% true negatives, and 1.2% wrong predictions.
### (4) Finding the support vectors of the trained SVM
The decision boundary is determined by “support vectors”<br/>
![](images/Figure_3.JPG)<br/>
### (5) Visualizing the decision boundary: by means of a cool extension called Mlxtend, we can visualize the decision boundary of our model. We’re going to show you how to do this with your binary SVM classifier.
evaluate your binary SVM classifier.
