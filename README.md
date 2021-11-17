# missing-data
Exploring solutions for imputing missing data for data analysis

As social scientists, we often run into issues of missing data. This can hinder the development of insights from data. So far, we have only come across basic solutions to matrix completion (often called imputation). Taking the mean, median, or mode for each feature is a common solution or potentially building a machine learning model to predict each feature based on other features.

Nguyen et al. (2019) identifies three methods to use when the lowest potential rank of an incomplete matrix is unknown, all through the method of minimizing the nuclear norm: (1) solving directly through gradient descent, (2) Singular Value Thresholding, and (3) Iteratively reweighted least squares.

We investigate:
●	How do the different matrix completion algorithms differ?
●	How much missingness can the matrix have and still converge to the minimum nuclear norm solution?
●	Does conditionally missing data differ from missing-at-random, with respect to some features that are particularly sparse? For example, a specific age demographic might not report income.
●	How well does each approach approximate the original data in terms of the sum squared distance over each element in the matrix after normalization? We could also consider a simple least squares model to test how different approaches affect predictive power.
●	How robust is matrix completion to a range of initial ranks? We could simulate data to test full datasets that have very high rank or very low rank.
