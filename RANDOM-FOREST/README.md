# Random Forest:
Random forest is a commonly-used machine learning algorithm trademarked by Leo Breiman and Adele Cutler, which combines the output of multiple decision trees to reach a single result. Its ease of use and flexibility have fueled its adoption, as it handles both classification and regression problems. you can study Decision tree in decision tree project readme file .

## My code :
I used the iris dataset.you have to bagging data to develop a random forest (RF).
### bagging :
Bagging happens randomly on features as well as on samples.
For example, from a 100x10 data, you can create a 70x5 size data with the number of decision trees that the examples and characteristics of the data made by Sedeh have been chosen randomly and without repetition or permutation. After this step, we have a data for each tree, which is a subset of the original data.

### Train and Prediction(test) :
I used the decision tree class for training data. And for each data and each tree, a model was obtained and I saved it.
To evaluate the obtained models, I considered a fixed data set from the original data set and tested the model only on the vignettes that had been trained.
And now, for each model, there is a vector of labels predicted by each decision tree, which I stored in a table.

### Decision tree evaluation:
After the above steps we have a presentation in which each line corresponds to the predictions of a model or a tree and to get the final prediction of the random forest from the column form of the majority voting method.i have used And then
i have measured the final accuracy of our random forest model.


