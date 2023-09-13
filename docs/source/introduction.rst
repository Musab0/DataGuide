1. Introduction
=====

Artificial intelligence (AI) systems have become a reality and affect our lives in many important ways. Data are at the core of artificial intelligence and machine learning (AI-ML) models; they are the main resource that enables AI-ML models to learn and evolve, allowing them to solve classification, prediction and anomaly detection tasks. Collecting, preparing and managing the data assets needed to train and deploy effective AI-ML models is a challenge. It is important that performance is achieved while making sure that AI-ML models abide by data protection regulations. Data usage is a fundamental aspect to consider when AI-ML systems are designed, implemented and operationalised. Being aware of data management problems allows organisations and individuals to understand and follow how their data are collected, transmitted, stored, processed and exploited by AI-ML- powered systems. Anyone who plans to implement such systems should be fully aware of all challenges related to data management in order to ensure consistent and sustainable AI-ML deployment.

This document is for data owners and organisations wishing to adopt and deploy AI-ML models using these data. It discusses how data assets are used in the AI-ML models’ life cycle and highlights the best practices for using them.

Throughout the document, the term data will be used to define the representation of facts, measurements and various other types of information, while the term data asset will designate the incarnation of data in a format that is suitable for management, storage and processing within information and communication technology (ICT) systems. The document presents data management practices starting from the life cycle of AI-ML models that rely on such data. Indeed, one of the most common pitfalls is to use data that are not suitable for a given stage of AI-ML model development, or data that are not suitable at all, and then expect reasonable performance. It is important to realise that increasing data quantity does not usually mitigate the problem of having low-quality data. As an introduction to the overall data landscape, the next section provides a short introduction to data types. Then the chapter describes a reference life cycle for AI-ML models, which will be used throughout the document to map data assets to AI-ML models’ development stages, explain the role of the data assets and describe the best practices for their collection and management.


1.1	Types of Data
-------
Data can be classified into two main categories, based on the way they are organised. The first category is structured data, which is data formatted in such a way that each data item has the same standard, predefined structure, usually described via metadata.

Structured data are easy to index, search and manipulate due to their predefined structure. Typical examples of structured data are relational databases, where data entities (e.g., the employees of a company) are represented by tables whose rows correspond each to a data item (the individual employee), while columns represent attributes (e.g., each employee’s name, age, gender). Attributes belong to elementary data types [#f1]_ (e.g., integer, date, string and timestamps). The structure of each table is described by an essential metadata item, called the database’s schema.

The second data category is unstructured data, where each data item can have a different format and size, without a clear predefined structure. Un- structured data are usually more difficult to deal with, because they require preprocessing to extract key information from them. Examples of unstructured data range from text documents and web pages to audio recordings. Advanced techniques such as natural language processing (NLP) allow for the extraction of key information from unstructured text.

For both data categories, we can distinguish between numerical and categorical attributes. This distinction is of capital importance for data representation and for later processing.


.. rubric:: Footnotes

.. [#f1]  Besides elementary data types like strings and integers, structured data items can also belong to complex types, like the nested records used for log file entries. 

1.2	DATA QUALITY
---------
A well-known saying in the realm of computing is ‘garbage in, garbage out’, meaning that if you feed any system or algorithm with low-quality data, the output will also be of low quality. AI-ML systems are no exception: they are not able to magically extract valuable insight from low-quality data, nor can they perform well without large quantities of high-quality data. Let us now list and describe some types of low-quality data that, when fed to AI-ML models, usually result in poor-quality output. 

* Redundant, Outdated or Trivial (ROT) data. . It is very easy and common to duplicate data or have the same (or similar) data collected by multiple entities. While the size of data is an important factor, redundant data are not useful for AI applications. Having diverse and comprehensive data is much more effective for AI-ML applications than a large set of redundant data. 

* Dark Data. This term refers to data collected and stored by an organization but never used. Often, data ends up unused because it is collected without a clear purpose, clobbering up the ICT infrastructure of organizations that do not have the proper skill level to use the data or cannot make the resource and time investment to tap into it. Other times, dark data originate from data collection procedures carried out carelessly, without upholding any collection best practice or standard. Dark data do not just lay dormant: they can become harmful to organizations. Indeed, storing dark data and maintaining the platforms that host them may have a non-negligible cost. While there is no clear answer on how to deal with dark data, simply deleting them can often save organizations time and money. 

Generally speaking, it is possible to consider data quality as a series of dimensions describing the quality of the information fed to (and produced by) an AI-ML model; that is, a measure of the success of the system utilizing this information. Therefore, both input and output data can be considered products with a certain quality. We now list a few aspects of data quality that should be pursued when selecting raw data. The most common ones are completeness, consistency, fitness for use, relevance and timeliness. For the purposes of this document, two additional aspects of data quality are significant: uncertainty and vagueness, which can be seen as two different aspects of indeterminacy, i.e. how much it is known about the consequences of exchanging a data item. Uncertainty is mainly related to the error or imprecision associated with raw data, while vagueness is an inherent issue of categorical values (for example, in the sentence “long text”, how many characters does “long” mean?). In the case of information expressed as text, one can distinguish between uncertainty due to the writing style (imprecision, vagueness, polysemy) and uncertainty due to the text content (for example “Mary gave Sally her book”). 


1.3	THE AI-ML LIFE CYCLE
-------
We are now ready to discuss the different data assets generated and used by AI-ML applications. Our discussion will be driven by a basic notion of systems engineering: the development life cycle, which is used to designate the process of planning, developing, testing and deploying an information system. The AI-ML applications life cycle (in short, the AI-ML life cycle) defines the phases that organisations follow to take advantage of supervised machine learning (ML) models to derive practical business value. Most of these stages use and/or generate specific data assets, whose careful management is the goal of this document. The AI-ML life cycle covers only a part of the AI applications landscape; other types of AI models will be discussed in Section 9. Figure 1.3 shows the different stages of the AI-ML life cycle. 

In this chapter, we provide a short definition of each stage and outline the individual steps it involves (‘Phase in a Nutshell’). For the sake of clarity, we also present an instance of each stage within the framework of a running example concerning a sample AI-ML application. We start by providing a general description of the running example. Then, for each phase of the AI- ML life cycle, we will provide a short description of the phase in the context of our running example (under the title ‘Phase in Our Running Example’). This description should help the reader to understand which data assets are concretely needed at each phase and how they are used. 


1.3.1 A Running Example for the AI-ML Life cycle
~~~~~~~~~~~
The ACME oil field services company wants to prevent the failure of its mechanical equipment. ACME uses a high-speed rotating machine (internally called a type-A rotatory) to mix components with water to make a frothy mix used to produce shale gas. Rotating machines of type A run for weeks without interruptions, leading to frequent breakdowns. The need to find a solution to predict failure of the equipment is dire, since it is a critical component for oil and gas exploration. ACME intends to develop a AI-ML model called a binary predictor [#f2]_ that will run continuously and assign to each ACME rotating machine a label regarding the next failure (either IMMINENT or NOT-IMMINENT). Machines labelled IMMINENT are to be immediately stopped for maintenance in the hope that their downtime due to maintenance will be shorter than the downtime that would result from a breakdown. The performance of the AI predictor will be validated by comparing the total downtime with the AI predictor in operation to downtime without the predictor, obtained from historical data. Any change (positive or negative) observed when using the AI predictor will indicate the performance gain or loss. 

1.3.2 Business Goal Definition
~~~~~~~~~~~
Before carrying out any development or deployment of AI applications, it is important that all stakeholders fully understand the business context of the AI application and the data required to achieve the AI application’s business goals, as well as the business metrics to be used to assess the degree to which these goals have been achieved. 

.. admonition:: Business Goal Definition Phase in a Nutshell

   Identify the business purpose of the AI-ML model. Link the purpose with the question to be answered by the AI model. Identify the model type based on the question. 
   Business Goal Definition in Our Running Example: Using a standard technique for management decisions like the goal-question-metrics approach, ACME management can specify the business objectives of the planned AI application as follows. Goal: Decrease the downtime of rotating machines of type A. Question: Is predictive maintenance of type-A equipment before its (estimated) failure time more cost- and downtime-effective than reactive maintenance af- ter breakdown? Metrics: The total cost of operation for type-A equipment. 

1.3.3 Data Ingestion
~~~~~~~~~~~
Data ingestion is the AI life cycle stage where data are obtained from multiple sources to compose data records, for immediate use or for storage in order to be accessed and used later. Data ingestion lies at the basis of all AI applications. Data can be ingested directly from their sources as they are generated (streaming) or via periodically importing blocks of data called batches. Indeed, stream and batch data ingestion can be active in the same AI application simultaneously. For example, the licence plates of cars entering a parking lot can be ingested one by one to check them against a stolen cars database, while batches of the same data are collected periodically for computing the parking lot’s average occupancy. An important data management procedure performed at ingestion time is data filtering or access control. This procedure selects data to be ingested, depending on their privacy status (personal/non-personal data, consent given for a given purpose, etc.). We will deal with these issues in detail in Sect. 1.7. For now, we only remark that it is good practice at ingestion time to apply some anonymity preservation techniques, taking into account the achievable trade-off between the impact of potential disclosure and the accuracy of the analysis to be computed on the data [#f3]_. 

.. admonition:: Data Collection/Ingestion Phase in a Nutshell

   Identify the input data to be collected and the corresponding annotation metadata. Organise ingestion according to the AI application requirements, importing data in a stream, batch or hybrid fashion. 
   Data Collection/Ingestion Definition in Our Running Example: In the fault prediction application for rotating machines, a stream of sensor data must be ingested about the operation of each rotatory (serial number, working conditions [round/ min], input power [kw], input mass [kg], output). Batch ingestion is also needed (usually via a separate database query) for the corresponding context (meta) data: equipment brand, model, serial number, procurement info (supplier, date of construction, date of delivery), installation data (installer, date of 


.. rubric:: Footnotes

.. [#f2] Besides elementary data types like strings and integers, structured data items can also belong to complex types, like the nested records used for log file entries. 
.. [#f3] For multimedia data sources, access control rather than being based on filtering may follow a digital rights management approach where some proof-of-hold are negotiated with the data owner’s license servers before ingesting the data. 

1.4 Data Exploration
-------
Data exploration is the stage where insights start to be taken from ingested data. While this stage may be skipped in some AI applications where data are well understood, it is often a crucial (and very time-consuming phase) of the AI-ML life cycle. At this stage, it is critical to distinguish between numerical and categorical data. Numerical data lends itself to plotting and allows for computing descriptive statistics and verifying if data fit simple parametric distributions like the Gaussian one. Missing data values can also be detected and handled at the exploration stage. 

.. admonition:: Data Validation/Exploration in a Nutshell

   It is always advisable to plot data after ingestion, to obtain a multidimensional view of all the components of each data vector. Also, it is useful to verify if data fit a known statistics distribution, either by component (monovariate distribution) or as vectors (multivariate distribution), and estimate the corresponding statistic parameters. 
   Data Validation/Exploration in Our Running Example: ACME data scientists will periodically plot sensed data about multiple pieces of equipment (e.g., the rounds-per-minute and power consumption variables) and fit the data to a bivariate statistical distribution (e.g., a Gaussian or power-log distribution). If the statistical tests confirm data belong to a distribution, they will display the distribution’s parameters, for instance the standard deviation σ, and highlight ‘three-sigma’ outliers (e.g., the machines whose rotation speed values lie outside an interval of three sigmas around the average). 

1.4.1 Data Pre-Processing
~~~~~~~~~~~
Data preprocessing can be the most critical stage of the life cycle. At this stage, techniques are employed to clean, integrate and transform the data, resulting in an improved data quality that will save time during the analytic models’ training phase and promote better quality of results. Data cleaning is used to correct inconsistencies, remove noise and anonymise data. Data integration puts together data coming from multiple sources, while data transformation prepares the data for feeding an AI-ML model, typically by encoding it in a numerical format. A typical encoding is one-hot encoding used to represent categorical variables as binary vectors. This encoding first requires categorical values to be mapped to integer values. Then, each integer value is represented as a binary vector that is all zero values except the position of the integer, which is marked with a 1. Figure 4 below shows one-hot encoding of categorical data expressing colours. 

Once converted to numbers, data can be subject to further types of transformation: rescaling, standardisation, normalisation and labelling. Rescaling expresses numerical data in a suitable representation unit (e.g., from tons to kilograms). Standardisation puts data in a standard format, and normalisation maps data to a compact representation interval (e.g., the interval [0, 1], by dividing all values by the maximum). Labelling (done by human experts or by another AI application) associates each data item to a category or a prediction. At the end of this process, a numerical data set is obtained, which will be the basis for training, testing and evaluating the AI model. 

.. admonition:: Data pre-processing in a Nutshell

   Convert ingested data to a metric (numerical) format, integrate data from different sources, handle missing/null values by interpolation, increase density to reduce data sparsity, de-noise, filter out outliers, change representation interval. Anonymize the data. 
   Data Preprocessing in Our Running Example: After having ingested the sensor data about the rotating machines, the ACME AI-ML application interpolates any missing value about equipment rotation speed and power consumption to achieve a uniform samples/time unit rate. The application integrates sensed data about rotation speed and power with data about external temperature and atmospheric pressure at the same time obtained from an open data service; then, it normalizes the data vectors, and adds to each data vector labels IMMINENT NOTIMMINENT representing the expected time to next failure. Also, it deletes the human operator code from the data to make sure they do not reference personal information. 

1.5 Feature Selection
-------
Feature selection is the stage of the life cycle where the number of components of the data vectors (also called features or dimensions) is reduced, by identifying the components that are believed to be the most meaningful for the AI model. The result of this phase is a reduced data set, as each data vector has fewer components than before. Besides the computational cost reduction, feature selection can help in obtaining more accurate models. Additionally, models built on top of lower dimensional data are more understandable and explainable. This stage can also be embedded in the model-building phase, to be discussed in the next section. 

.. admonition:: Feature selection in a Nutshell

   Identify the dimensions of the data set that account for a global parameter (e.g., the overall variance of the labels). Project data set along these dimensions, discarding the others. 
   Feature Selection in Our Running Example: In the predictive maintenance application, the ACME data scientists project the vectors of the data set on the subset of dimensions that maximises input variance [#f4]_. As inputs are mostly numerical data (like the engines’ power consumption and rotation speed), ACME data scientists use the principal component analysis (PCA) method. If inputs had been categorical, multiple correspondence analysis could have been used to represent categorical data as points in a low-dimensional vector space. 

.. rubric:: Footnotes

.. [#f4]  Besides elementary data types like strings and integers, structured data items can also belong to complex types, like the nested records used for log file entries. 

1.5.1 ML Model Selection
~~~~~~~~~~~
This stage performs the selection of the best AI-ML model or algorithm for analysing the ingested and preprocessed data. Finding the ‘right’ AI-ML model to solve a business problem or achieve a business goal is a challenge, often subject to trial and error. Based on the business goal and the type of available data, different types of AI techniques can be used. It is important to remark that model selection may trigger a transformation of the input data, as different AI models require different numerical encoding of the input data vectors. Two major categories are supervised learning and unsupervised learning models, the latter including clustering and reinforcement learning. Supervised techniques deal with labelled data: the AI-ML model is used to learn the mapping between input examples and the target outputs. Supervised models can be designed as classifiers, whose aim is to predict a class label, and regressors, whose aim is to predict a numerical value function of the inputs (e.g., a counter). Unsupervised techniques extract relations from unlabelled training data, with the aim of organising them into groups (clusters, highlighting associations among data, summarising data distribution and reducing data dimensionality [this last already mentioned as a goal of data preparation]). 

Reinforcement learning is typically less data dependent: it maps situations with actions, learning behaviours that will maximise a reward. 

AI-ML models of different types can be composed using composition methods (e.g., by taking the majority of their outputs)[#f5]_. 

.. rubric:: Footnotes

.. [#f5]  Besides elementary data types like strings and integers, structured data items can also belong to complex types, like the nested records used for log file entries. 

.. admonition:: AI Model Selection in a Nutshell

   Choose the type of AI model most suitable for the application. Encode the data input vectors to match the model’s preferred input format. 
   AI Model Selection in Our Running Example: For associating an IMMINENT or NOT-IMMINENT label to each data vector about the type-A rotating machines, ACME data scientists choose a multidimensional, supervised AI model with memory, as they realise that fault events depend on the history of each piece of equipment and not only on the current values of the input. They choose a two-dimensional long short-term recurrent neural network (2D RNN). They compute one-hot encoding of the categorical inputs and map the input data vectors (dimension n) into 2D tensors (i.e., bi-dimensional matrices with dimensions h, k and h + k = n). 

1.5.2 Model Training
~~~~~~~~~~~
When the selected AI analytic is an ML model, the latter must go through a training phase, where internal model parameters like weights and bias are learned from data. The training phase will feed the ML model with batches of input vectors and will use a learning function to adapt the model’s internal parameters (weights and bias) based on a linear or quadratic measure of the difference between the model’s output and the labels. Often, the available data set is partitioned at this stage into a training set, used for setting the model’s parameters, and a test set, where error is only recorded in order to assess the model’s performance outside the training set. Cross-validation schemes randomly partition the data set multiple times into a training and a test portion of fixed sizes (e.g., 80% and 20% of the available data) and then repeat training and validation phases on each partition. 

.. admonition:: AI Model Training in a Nutshell

   Select and apply a training algorithm to modify the chosen model according to training data. Validate the model training on test set according to a cross-validation strategy. 
   AI Model Training in Our Running Example: Train the 2D RNN model for type-A equipment failure prediction via a small batch gradient descent algorithm with L2 loss function on the training set. Use the 80-20 cross-validation strategy. 

1.5.3 Model Tuning
~~~~~~~~~~~
Certain mathematical parameters define the high-level behaviour of ML models during training, such as the learning function or modality mentioned above. It is important to know that these parameters, often called hyperparameters, cannot be learned from input data. They need to be set up manually, although they can sometimes be tuned automatically by searching the model parameters’ space, in practice by repeatedly training the model, each time with a different value of hyperparameters. This procedure is called hyperparameter optimisation. It is often performed using classic optimisation techniques like grid search, but random search and Bayesian optimisation can also be used. 

For the purposes of this document, it is only important to remark that the model tuning stage uses a special data asset (often called a validation set), which is distinct from the training and test sets we described in the previous stages. Also, it is useful to know that a final evaluation phase (after tuning) is sometimes carried out to estimate how the tuned model would behave in extreme conditions, for example, when fed with wrong/unsafe data sets. The extreme data used for the latter procedure is called held-out data. 

.. admonition:: AI Model Tuning in a Nutshell

   Apply model adaptation to the hyperparameters of the trained AI model using a validation data set, according to deployment condition. 
   AI Model Tuning in Our Running Example: ACME data scientists run the 2D RNN model they trained for fault prediction on an additional validation data set and choose the best values h and k for the RNN’s tensor dimensions. Then they es- timate how the tuned model would behave in extreme conditions by running the model on some held-out data corresponding to extreme rotation speed values.

1.5.4 Transfer Learning
~~~~~~~~~~~
