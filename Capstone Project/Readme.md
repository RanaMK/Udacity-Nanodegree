
# Udacity Data Scientist Capstone Project -- Starbucks Capstone Challenge

## Medium Blog Post

Here is the Medium blog post I have written:

https://rana-mahmoudkamel.medium.com/udacity-data-scientist-capstone-project-a037eb661a5


## Project Overview
In this capstone project, I am using all the knowledge I have learned in the Udacity Data Scientist Nanodegree.

I chose the "Starbucks Challenge" as my final project where data was collected by Arvto and has 3 main datasets. The first one contains the customers data, the second contains the offers data and the third contains the event log for customer's purchase.

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set.

The main task of the project is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

The given dataset contains transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer.

Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.


## Problem Statement

The program used to create the data simulates how people make purchasing decisions and how those decisions are influenced by promotional offers.

Each person in the simulation has some hidden traits that influence their purchasing patterns and are associated with their observable traits. People produce various events, including receiving offers, opening offers, and making purchases.

As a simplification, there are no explicit products to track. Only the amounts of each transaction or offer are recorded.
There are three types of offers that can be sent: buy-one-get-one (BOGO), discount, and informational. In a BOGO offer, a user needs to spend a certain amount to get a reward equal to that threshold amount. In a discount, a user gains a reward equal to a fraction of the amount spent. In an informational offer, there is no reward, but neither is there a requisite amount that the user is expected to spend. Offers can be delivered via multiple channels.

The basic task is to use the data to identify which groups of people are most responsive to each type of offer, and how best to present each type of offer.
So in this project, we will use machine learning to predict the response of customers to offers either by "offer received", "offer viewed" or "offer completed". 
This information will be predicted based on some demographic information of the users as well as other purchasing data.

In this project, I followed the CRISP-DM process in the notebook code using the below steps:

- Business Understanding
- Data Understanding
- Prepare Data
- Data Modeling
- Visualize the Results

## Files in this repository
* data folder -- contains 3 json files
  * portfolio.json -- information about the offers provided by Starbucks
  * profile.json -- demographic data of the customers
  * transcript.json -- transactions made by the customers including offers received, viewed and completed
* Starbucks_Capstone_notebook.ipynb -- python code for the project


## Datasets Description
* profile.json
Rewards program users (17000 users x 5 fields)
  * gender: (categorical) M, F, O, or null
  * age: (numeric) missing value encoded as 118
  * id: (string/hash)
  * became_member_on: (date) format YYYYMMDD
  * income: (numeric)

* portfolio.json
Offers sent during 30-day test period (10 offers x 6 fields)
  * reward: (numeric) money awarded for the amount spent
  * channels: (list) web, email, mobile, social
  * difficulty: (numeric) money required to be spent to receive reward
  * duration: (numeric) time for offer to be open, in days
  * offer_type: (string) bogo, discount, informational
  * id: (string/hash)

* transcript.json
Event log (306648 events x 4 fields)
  * person: (string/hash)
  * event: (string) offer received, offer viewed, transaction, offer completed
  * value: (dictionary) different values depending on event type
      * offer id: (string/hash) not associated with any "transaction"
      * amount: (numeric) money spent in "transaction"
      * reward: (numeric) money gained from "offer completed"
  * time: (numeric) hours after start of test


## Installed Python Libraries

I am using Python3 with the below libraries:
* pandas
* numpy
* matplotlib
* math
* json
* sklearn
* seaborn


## Evaluation Metrics
The performance of machine learning algorithms can be measured with four main metrics.

Before explaining the metrics, there are some terminologies to be described:

* True Positives : the number of correctly classified observations as 'Yes' or positive.
* True Negatives : the number of correctly classified observations as 'No' or negative.
* False Positives : the number of wrongly classified observations as 'Yes' or positive while it should be 'No'.
* False Negatives : the number of wrongly classified observations as 'No' or negative while it should be 'Yes'.

Let's now explain the most important metrics in machine learning as follows.

* Accuracy: it is the most intuitive performance measure. It is calculated by dividing the True Positive + True Negative by True Positives + True Negatives + False Positives + False Negatives.
* Precision: it focuses on the ratio of correct positive observations.
* Recall: it is also known as sensitivity or true positive rate. It is the ratio of correctly predicted positive events.
* F1-score: it takes into account the precision and recall to compute the model performance and it is called the harmonic rate of Precision and Recall.
* Confusion Matrix: it is another way to calculate any of the metrics above and specially used while running codes. It is a matrix showing the True Positive, True Negatives, False Positive and False Negative counts. The accuracy can be calculated by taking average of the values across the main diagonal.

Accuracy is used when the True Positives and True negatives are more important while F1-score is used when the False Negatives and False Positives are more important.
In classification problems, imbalanced class distribution exists and thus F1-score is a better metric for evaluation that's the reason we used F1 score as evaluation metric for this project.


## Improvements
We could try other classifiers to improve the model accuracy. Also, we could try to use deep learning instead of machine learning and check if this could impact the results.

Another improvement could be done is to balance the data by extracting an equalnumber of records for each event type "offer received", "offer viewed" and "offer completed". For example, deleting "offer received" and "offer viewed" rows for completed offers.


## Conclusion of Analysis
In this project, we used machine learning to predict the purchasing type of users based on customer's properties as well as other purchasing attributes. We used 3 versions for running the model using 3 different classifiers:
- Random Forest
- K Nearest Neighbors
- Decision Tree

We can see that the three models perform almost the same with an accuracy between 74 ad 76% which is acceptable.

There is one comment on data imbalance as most of the "offers" have purchasing type "offer received" and not "viewed" or "completed" as shown in the above tables of the counts of predicted target values from all models so most of the events are predicted as "offer received" because "offer received" is the most occuring event.

## Licensing, Authors, Acknowledgements .
Data for coding project was provided by Udacity.
