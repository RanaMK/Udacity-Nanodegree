# Writing Data Science Blog Post

## Medium Blog Post

Here is the Medium blog post I have written:

https://rana-mahmoudkamel.medium.com/how-working-with-programming-languages-change-over-the-past-4-years-52f361ba86be

## Project Motivation

This project (Writing a Data Science Blog Post) is part of Udacity Data Scientist Nanodegree Program.

I used the Stackoverflow Developer Annual Survey for the years 2017, 2018, 2019 and 2020.

The original Dataset can be found here:https://insights.stackoverflow.com/survey
I used the Survey csv for each year.

In this project, I focus on the development field and software engineering as a lot of people and students would like to know what are the current and future status of programming.
The first question that comes to mind when talking about programming and development is what are the top used programming languages and are those languages going to disappear in the future or not so we are going to check the top programming languages as well as database languages used by actual developers and what are their thoughts about languages in the future. 

The code and blog post is divided into 3 main questions answered through the data.
Question 1: What are the top programming languages used in the last four years wordwide?
Question 2: How does working with databases change over time?
Question 3: How developers are thinking about programming and database languages in the future?

To answer th above questions, I followed the CRISP-DM process in the notebook code using the below steps:
* Business Understanding
* Data Understanding
* Prepare Data
* Data Modeling
* Visualize the Results

## Libraries

I am using Python3 with no special libraries.
The libraries I used: Numpy, Pandas,Matplotlib and Seaborn.

## Files in this repository
Working with programming and DB languages Code.ipynb -- containing the code for extracting and analyzing data in addition to some visualizations to best understand the data

## Conclusion of Analysis
In this project, we took a look at the Stackoverflow Annual Survey data from 2017 to 2020 to make an analysis on the top used programming and database languages by actual developers. In addition, we checked from their response, what are their desired working language in the future. Finally, we summarised the data by checking the common used and desired languages as well as the languages appealing and not appealing for developers to work with in the future.

For question one, we can see that the top programming languages in the past 4 years didn’t change a lot and the main languages used by developers are Java, Python and Javascript as well as web development with HTML/CSS and the main database language SQL.

For question2, we can see that starting 2017, some non relational databases appeared like PostgreSQL and MongoDB. However, the relational databases are still appearing till 2020 such as MySQL, SQL Server, SQLite, Microsoft SQL Server…etc. From this result, we can see that although the world is heading to use Non-relational database more for its additional functionalities but the traditional relational databases are indispensable.

And finally for question 3, we can see that most of the desired programming and database languages are already the top current used languages by developers which makes us look deeper into the data and check the common languages that each developer use and what are the languages that are currently used by the developers and want to get rid of it or what are the new languages that they want to work with.

Please take a look at the article or the code to see more about this analysis and check the visualizations for better understanding.

This analysis is not covering all the programming and database languages used in the whole world and is only focusing on the data provided by developers who answered the Stackoverflow annual survey.
