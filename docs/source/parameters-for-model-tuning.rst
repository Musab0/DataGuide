6. Parameters for Model Tuning
===========
There are certain parameters which define high level concepts relating to ML models, such as their learning function or modality, and cannot be learned from input data. These model parameters, often called hyper-parameters, need to be set manually, although they can be tuned automatically by searching the model parameters’ space. This search, called hyper-parameter optimization, is often performed using classic optimization techniques like Grid Search, but Random Search and Bayesian optimization can be used. It is important to remark that the Model Tuning stage uses a special data set (often called validation set), distinct from the training and test sets used in the previous stages. An evaluation phase can also be considered to estimate how the model would behave in extreme conditions, for example, by using wrong/unsafe data sets. 


6.1 Model Hyper-Parameters
-----------

6.2 Hyperparameters Optimisation Strategies
-----------

6.3 Transfer Learning
-----------
