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

.. [#f1] Text of the first footnote.

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

.. admonition:: Business Goal Definition Phase in a Nutshell:

Identify the business purpose of the AI-ML model. Link the purpose with the question to be answered by the AI model. Identify the model type based on the question. 
Business Goal Definition in Our Running Example: Using a standard technique for management decisions like the goal-question-metrics approach, ACME management can specify the business objectives of the planned AI application as follows. Goal: Decrease the downtime of rotating machines of type A. Question: Is predictive maintenance of type-A equipment before its (estimated) failure time more cost- and downtime-effective than reactive maintenance af- ter breakdown? Metrics: The total cost of operation for type-A equipment. 

1.3.3 Data Ingestion
~~~~~~~~~~~
Data ingestion is the AI life cycle stage where data are obtained from multiple sources to compose data records, for immediate use or for storage in order to be accessed and used later. Data ingestion lies at the basis of all AI applications. Data can be ingested directly from their sources as they are generated (streaming) or via periodically importing blocks of data called batches. Indeed, stream and batch data ingestion can be active in the same AI application simultaneously. For example, the licence plates of cars entering a parking lot can be ingested one by one to check them against a stolen cars database, while batches of the same data are collected periodically for computing the parking lot’s average occupancy. An important data management procedure performed at ingestion time is data filtering or access control. This procedure selects data to be ingested, depending on their privacy status (personal/non-personal data, consent given for a given purpose, etc.). We will deal with these issues in detail in Sect. 1.7. For now, we only remark that it is good practice at ingestion time to apply some anonymity preservation techniques, taking into account the achievable trade-off between the impact of potential disclosure and the accuracy of the analysis to be computed on the data [#f3]_. 

.. admonition:: Data Collection/Ingestion Phase in a Nutshell:

Identify the input data to be collected and the corresponding annotation metadata. Organise ingestion according to the AI application requirements, importing data in a stream, batch or hybrid fashion. 
Data Collection/Ingestion Definition in Our Running Example: In the fault prediction application for rotating machines, a stream of sensor data must be ingested about the operation of each rotatory (serial number, working conditions [round/ min], input power [kw], input mass [kg], output). Batch ingestion is also needed (usually via a separate database query) for the corresponding context (meta) data: equipment brand, model, serial number, procurement info (supplier, date of construction, date of delivery), installation data (installer, date of 


.. rubric:: Footnotes

.. [#f2] Besides elementary data types like strings and integers, structured data items can also belong to complex types, like the nested records used for log file entries. 
.. [#f3] For multimedia data sources, access control rather than being based on filtering may follow a digital rights management approach where some proof-of-hold are negotiated with the data owner’s license servers before ingesting the data. 
