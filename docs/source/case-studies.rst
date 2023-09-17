8. Case Studies
===========

8.1 Case Study 1: Automatic Detection of Traffic Incidents
-----------
Traffic is known to be one of today’s significant issues affecting large cities and one of the main challenges for smart cities. Reducing delayed notification time for traffic incidents has a direct impact on reducing the fatality rate and reducing the cost of roadside assistance. Traffic-related data are being collected en masse from traffic sensors deployed in big cities and, more recently, from social media like X and Weibo. As such, automatic detection of traffic incidents has attracted much interest from traffic control centres over the last decades. The goal is to detect the occurrence of an event causing traffic congestion, including recurrent and non-recurrent congestion. This interest is driven by the need to develop automated inci- dent detectors – an asset in traffic management, as they can make appropriate and timely decisions based on the analysis of collected real-time data. 

8.1.1 Data ingestion
~~~~~~~~~~~~
The data used in the case study are at the macroscopic level; they are stored in a data hub which receives sensors’ data from a road section containing multiple lanes going in the same direction (i.e., a road link) and averages the traffic variables for every two minutes. Since the data are aggregated, there are no privacy issues at the individual level. Table 1 shows an example of a reading stored in the data hub. 

A road link is positioned between two points in the road and has a single direction, clockwise or anti-clockwise, and the distance each road link covers is different. Junctions may contain one or more links, each one covering a short distance, while road links connecting junctions cover a longer distance. 

8.1.2 Data exploration and pre-processing
~~~~~~~~~~~~~
The data received from the data hub still have certain limitations. There are periods where some road links have missing readings, making the detection of events rather difficult. Also, some readings contain missing data (e.g., occupancy data tend to be missing more often than speed data). Missing values are, however, typical of reproduction (as opposed to synthetic or lab-based) sensor networks. 

In preprocessing, infrequent missing values are filled in by averaging between the preceding and the following readings. Detection from locations with an increasing number of missing values is not considered. 

We consider automatic event detection a classification problem, which requires labelled data to train the ML model. A list of officially reported traffic events during a single period could be used to provide the ground truth for model training. Therefore, reported events from the period of the traffic readings are collected. Any event that had no effect on road conditions is discarded, as was any event whose readings were irregular or unavailable. For each reported event, the area where it took place (encompassing multiple road links), a single starting time and expected end time are specified. 

Moreover, it is not feasible to use the full data set for training and testing, as most of the data set will contain no events. A common practice in the field of traffic incidents detection is considering the readings of a certain number of hours (e.g., two hours) before and after an event. This will ensure capturing all the events and the different variations from non-event readings. 

We remark that events include – besides accidents – slow-downs due to traffic congestion; both start and end times are approximate. 

Regarding the event’s location, the road link that is most affected by the event is chosen. We get this information from the speed values along with the duration of the event. A manual check of all events used for training and testing is also conducted to adjust their timings. We remark again that using the reports without any adjustments could result in inaccurate training and test sets. This problem is commonly encountered in traffic data analysis and has been reported multiple times in the field. 

To ease the task of preprocessing the data set, we developed a tool to select the start and end time of an event visually. The tool shows a plot of speed values against time during the targeted period and, if available, a similar plot for road occupancy. 

**Data Types**

The dataset can be created as a temporal time series by taking traffic values (e.g., speed) in sequential time periods. Also, it can be created as a spatial sequence where the traffic values can be taken from neighboring road links at the same time. Another option is to combine both sequences. Part of the project was investigating the best approach.

**Time series length** 

One of the variables that should be set by us is the fixed length of the time series. In the AID domain, we receive readings in fixed periods, which provides us the ability to create a time series of any length. As presented in Figure 14, each time series is a continuation of the one preceding it with a sliding window of 1 which mimics a real time AID problem. By assuming a real time problem, we assume no knowledge of an event’s duration; hence, we are limited to selecting the most suitable length which provides the best performance. After selecting the most suitable length, there is no added value in using other lengths. 

8.1.3 Model Training, Deployment and Maintenance
~~~~~~~~~~
After obtaining data for training from the data hub, preprocessing them and adding the labels, it is possible to start training the ML model. A variety of algorithms can be used, and at this stage the data scientist can explore and test the various options and compare their results using specific criteria. The criteria included overall accuracy, incidents detection rate and false alarm rate. By comparing multiple algorithms, rotation forest did yield the best results, justifying its use to create the ML model. 

A system deployed in the cloud is made ready to accommodate the ML model by pulling the readings from the data hub, preprocessing the data and then feeding them to the model, which outputs the possibility of an event happening. The result of the ML model is exported to the data hub to be matched with the related reading. An external dashboard is also connected to the data hub to enable the traffic control centre to view the readings and any potential events. 

There is also a feedback mechanism to tell the system when the predictions were not accurate. The feedback is used for the periodic retraining of the model to increase the performance with time. 

8.2 Case Study 2: Social Media (X) Analysis
-----------
The growth in social media applications and the exponential rise in their use have led to the accumulation and availability of a huge number of social media data that cover a variety of topics that are important to the platforms’ users and beyond. Discussion topics on social media platforms can provide news, opinions, views, feedback and more on a wide range of subjects from politics and the economy to reviews and feedback on products and services. The opinions expressed and the sentiments associated with them can provide valuable insight into the feelings of communities and individual users. This information can be valuable to businesses, customer service and even government organisations that serve and support their communities. 

As a result, the harvesting and analysis of social media content such as X posts have become extremely valuable to many organisations to help them better understand their users and communities and indeed understand their sentiments and views, almost in real time. 

One of the first steps towards achieving this objective is to harvest the data; in this case, we will look particularly at X. The platform has a large online community where opinions and sentiments are expressed in as close to real time as possible. Anyone can have an account on X; many news agencies, journalists, politicians, TV broadcasters, businesses, government organisations and more use X as a means of communicating with their constituencies, users, customers and friends. Hence, X posts offer a huge opportunity to analyse and better understand certain communities. X enables us to detect their interest, as well as changes in that interest, allowing us to adjust for user, customer and community needs quickly. 

8.2.1 X posts harvesting
~~~~~~~~~~~
X offers APIs at different price points to enable the collection of X posts and additional related data, based on specific terms and conditions about how the data can be used and what controls and restrictions apply. The different types of developer account pricing provide different levels of access in terms of the amount of data that can be collected over a certain time period. There are also restricted but free methods to access a limited number of X posts. It is important to adhere to the X terms and conditions and use policies to avoid discontinuation of service or even additional actions against any misuse. 

Developers can write software tools referred to as harvesters, to collect relevant X posts and any additional information, such as author ID, reposts and location information. The X posts can be harvested by area, keyword or author, among others. The harvester can run periodically and collect data in either micro- or macro-batches depending on user requirements and the number of X posts being collected. Once the data are collected they can be stored in the relevant storage infrastructure. There are also rules and regulations that apply to the archiving and use of X posts. 

8.2.2 Classification
~~~~~~~~~~~
Classification of social media messages (such as X posts) is one of the most fundamental and useful data analysis techniques and can be used for many applications. Uses include, but are not limited to the following: 

*	Sentiment analysis on customer service. Can automatically classify short messages as positive, negative or neutral. This helps to automatically monitor sentiment changes about products and customer services. This sentiment analysis can also be more refined, classified to more detailed categories and providing richer information (e.g., social media messages classified as outraged, angry, upset, unhappy, neutral, happy, thankful, satisfied, excited, etc.). 

*	Message filter. Can automatically filter out irrelevant messages and ignore them. For example, if Company A would like all the social media messages relevant to it, a keyword-based search will return messages including those keywords. However, not all messages including these words are relevant to Company A. A filter will classify those messages into one of the two categories: relevant or irrelevant. 

*	Topic classification. Classifies all relevant social media messages we might want to know more about (e.g., whether they are talking about fault for services/connection, complaints about wrong bills, inquiries about new products, or recommendations to friends about the good service). This task can be done by message classification, as well. 

To realise classification, three stages are involved: training the model, testing the model and applying the model on the fly. 

*	**Model training** is the process to build up a classifier model using the training data. The training data must be tagged manually with target categories (e.g., positive or negative in sentiment analysis; relevant or irrelevant for a filter). During the model training process, the model extracts and learns the patterns from the tagged messages, and the trained model will be tested and used in the later stages. 

*	**Model testing** is the process that ensures the generated model from stage one satisfies our requirements, mainly in terms of accuracy (e.g., the model can classify 95% of the messages to the target categories correctly). This is done by applying the generated model to the testing data. The tested data need to be manually tagged, as well, to compare the model output (as category) with the target output (tagged manually). This testing process might happen iteratively together with the training process. We obtain a model from the training data and test it on the testing data. If the accuracy is lower than our expectation, we might need more training data to refine the model, or we might consider using another technique of classification to see whether other techniques can complete this classification task better. This iterative process continues until we reach a satisfactory accuracy. 

*	**Applying the model** means using the satisfied model on real-world data and generating classification results automatically. 

As we mentioned in the testing stage, a high accuracy might be our main objective to obtain a good classifier model. If we can not reach our expected accuracy, there might be two reasons. Either the training data is not enough in which case we need to increase the number of training data; or the classification technique we are using is improper to this application which we need to improve the classification technique. The outcome of the classification can be then be stored in the storage infrastructure as attributes of the original social media content (X posts) or any other format depending on the application. 

8.2.3 Visualisation
~~~~~~~~~~~~~
Once the analysis has been completed, the results of the analysis must be displayed in an easy-to-understand format that provides the required insight for the application. The simplest way to show results can be to merely show statistics summarising the analysis outcome, such as the number of users mentioning a company or product or the number of satisfied customers (positive sentiment) and unsatisfied customers (negative sentiment). For further, more advanced visualisation, one can develop their own front-end visualisation to display custom views or use one of the many available visualisation capabilities and tools – open-source or commercial. Existing visualisation tools enable the plotting of data such as time series or events or, indeed, using MAS and GIS systems. Visualisation tools can also have predefined and easy-to-use templates providing a plethora of options for almost every use. They also provide various customisation capabilities for more advanced requirements. Such tools are usually used for better understanding analysis, insight and decision support. 

