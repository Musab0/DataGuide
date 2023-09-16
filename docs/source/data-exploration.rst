3. Data Exploration
==========
It is important to rely on a data source only after having carefully examined the data it provides. The huge size of data sets used for AI applications makes it difficult for humans to inspect raw data vectors directly: there are just too many of them. Statistical and visual analysis tools play an important role in the data exploration phase of the AI life cycle. 

3.1 Statistical Analytics
----------
In this section, we outline the main statistical procedures carried out in the data exploration stage. 

3.1.1 Variables and Covariates
~~~~~~~~~~
In statistical terminology, the dimensions of each data vector that will be used by prediction or classification are called variables, while the other dimensions are called covariates. When ingesting data about a phenomenon, such as the rotation speed of the rotatory engines mentioned in Section 1, covariates can be constant (that is, fixed at ingestion time, e.g., the engine’s model) or change over time (for example, the engine’s physical location). The covariate studies carried out in the data exploration stage try to determine which covariates are helpful for the prediction of classification of the variable values. Such studies aim to achieve results analogous to the feature reduction stage of the AI life cycle. 

3.1.2 Size Determination
~~~~~~~~~~
In the data ingestion stage, it is often implicitly assumed that the data points in a data set are samples representative of a wider population. An important requirement to be checked at exploration time is to have data from an adequate number of instances. Whatever the AI technique that will be used, any estimate based on a small number of instances will be less reliable than one based on a larger number, and when statistic models are fitted to small data sets, the estimated impact of the covariates is too imprecise to give reliable answers. A rule of thumb is that even when simple regression models are used, at least ten data points need to be included in the data set for each dimension considered; otherwise, regression coefficients may become biased. Several books and software packages are available to assist the calculation of adequate sample sizes, and many general-purpose statistical packages also perform such calculations. 

3.1.3 Distribution Fitting 
~~~~~~~~~~
A major part of exploratory statistical analysis is probability distribution fitting, (i.e., checking that a given statistical distribution or model is an appropriate representation of the ingested data). Fitting involves computing the corresponding distribution parameters (e.g., the distribution mean and variance based on the data set parameters average and deviation). Knowing which probability distribution is a close fit to the ingested data set can be very helpful in the next phases of the AI-ML life cycle, such as model selection. Figure 11 depicts two common distribution functions, the Normal (or Gaussian) distribution and the Uniform distribution. Generally speaking, however, distribution fitting is not straightforward, and requires some background in statistics. Here, we only aim to present an overview of some of the major issues involved. 
The first step usually consists of guessing the appropriate statistical distributions to try on the data. Building the cumulative distribution function (CDF) of the data values requires listing the frequency of each data value in the data set. From this histogram, one can derive the probability distribution function (PDF) of the data. Educated guesses about the data distribution made at the data exploration stage usually take into account the presence or absence of symmetry in the data set. For example, data values whose frequencies lie symmetrically around a value may fit the different shapes of the symmetrical normal (or Gaussian) distribution, depending on mean μ and variance σ. Other symmetric distributions are the logistic distribution, the Cauchy distribution or the Student’s t-distribution. The latter is an example of a symmetric heavy-tailed distribution, meaning that the values farther away from the mean occur relatively more often. 
When large data values tend to be farther away from the mean than smaller ones, one has a distribution skewed to the right. Skewed distributions include the log-normal one, where log values of the data are normally distributed, the exponential distribution, the Pareto distribution and many others. When small data values tend to be farther away from the mean than the larger ones, the distribution is left-skewed, like the square-normal distribution (i.e., a normal distribution applied to the square of the data values). Of course, the true probability distribution of data may be different from the fitted distribution, as the ingested data may not be totally representative of the underlying phenomenon due to measurement error. Also, there is non-stationary behaviour: the occurrence of data in the future may deviate from the fitted distribution. In other words, a change of environmental conditions may cause a change in their probability of occurrence. 
To quantify how well a distribution fits the data, it is customary to use parametric tests, where the parameters of the distribution are computed from available data. Such tests are available in many statistical packages and libraries. It is also customary to transform right-skewed asymmetric data to fit symmetrical distributions (like the normal and logistic ones) by applying the logarithm or the square root to the data, or to fit a left-skewed distribution by computing the square values. Rais- ing data to a power p, one can try to fit symmetrical distributions to data obeying a distribution of any degree of skewness. This technique enhances the flexibility of probability distributions and increases their applicability in distribution fitting. Another popular technique is distribution shifting (i.e., replacing each raw data value V by V i = Vm , where Vw is the minimum value of V ). This replacement represents a shift of the probability distribution to the right, as Vm is negative. After completing the distribution fitting of V i, the corresponding values are computed as V = V i + Vm, which represents a back-shift of the distribution to the left. Distribution shifting augments the chances of finding a properly fitting probability distribution. 
It is also possible to fit two different probability distributions, one for the lower data range, and one for the higher. The ranges are separated by a break-point. The use of such composite probability distributions may be advisable when the data are collected under different conditions. 



3.2 Visual Analytics
----------

Visual analytics is an outgrowth of the fields of information visualisation and scientific visualisation that focuses on analytical reasoning facilitated by interactive visual interfaces. It can attack certain problems whose size, complexity and need for closely coupled human and machine analysis may make them otherwise intractable. It integrates machine analysis process, human cognition and perception and information visualisation to lead the researcher in the process of analysis. The main aim of visual analytics is to amplify the analyst perception by providing a visual representation of the data that results from the analysis process. The analyst can interact with both the information visualisation, by zooming and filtering, and the analysis process by choosing the analytics methods or changing attributes. In this context, the cognitive ability of the analyst is the key to building hypotheses and making decisions. 
Visual analytics seeks to blend techniques from information visualisation with techniques from computational transformation and analysis of data. Information visualisation forms part of the direct interface between user and machine, amplifying human cognitive capabilities in a few basic ways: 

*	by increasing cognitive resources, such as by using a visual resource to expand human working memory; 
*	by reducing the search space, such as by representing a large amount of data in a small space; 
*	by enhancing the recognition of patterns; 
*	by supporting the easy perceptual inference of relationships that are otherwise more difficult to infer; 
*	by perceptual monitoring of a large number of potential events; and 
*	by providing a manipulable medium that, unlike static diagrams, enables the exploration of a space of parameter values. 

These capabilities of information visualisation, combined with computational data analysis, can be applied to analytic reasoning to support the sense-making process. 

3.2.1 Visual Analytics Process
~~~~~~~~~~~
During the visual analytics process, the user alternates data visualisation and analysis of results by trying to gain insight and knowledge that up to that point has been hidden. Figure 12 illustrates the visual analytics process: ovals represent stages, and arrows represent transitions. The process is iterated in subsequent steps until the analyst is satisfied with the extracted knowledge: 

* Some data sets may require transformations such as integration, cleaning or normalisation before analysis may begin. 
* The analyst is typically given two options: 
  * First, to visualise the data and come up with a hypothesis or remodel data. 
  *	Second, to analyse the data and build models using data mining methods, then visualise it. 
* The analyst is part of the loop in both cases. For visualisation, they can zoom in/out in the diagram to build hypotheses. Besides, in the analysis processes, they can choose the method of analysis or change parameters to test them. 

3.2.2 Some Examples
~~~~~~~~~~~
Common features of visual analytics tools include the capability of data visualisation across a number of dimensions, a rich and user-friendly dashboard, the capability of integrating different data sources and sometimes the support for multi-user collaboration in the analysis. An example of a well-known analytics tool with multidimensional visualisation is Gapminder Trendalyzer. It can be described as a bubble chart using animation to illustrate trends over time in three dimensions: one for the X-axis, one for the Y-axis, and one for the bubble size, animated over changes in a fourth dimension (time). Colour and other graphical markings can add extra dimensions. For instance, one can represent the average income of people within a country on the X-axis, life expectancy on the Y-axis and the population as the size of a bubble, and use bubble colours to denote the continent where they are located. Using these conventions, one can observe the time evolution of the first three quantities over time and, for instance, make hypotheses about how the correlation between the first and the second has developed. Selecting one continent rather than another allows us to formulate hypotheses about the different dynamics present in distinct regions of the world. The tool is also effective for storytelling. Rich dashboards are featured by commercial products. They offer a wide selection of gauges, data views, maps, charts, widgets, tables and other data-aware objects for story boarding and data representation. 





