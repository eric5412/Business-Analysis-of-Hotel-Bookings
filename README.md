Business Analysis of Hotel Bookings
================


### Introduction:

The scope of this project was to analyze booking data for two hotels
located in Portugal. One hotel is a City Hotel and the other is a Resort
Hotel.

**Data Source:** Hotel Booking Demand dataset on
[Kaggle](https://www.kaggle.com/jessemostipak/hotel-booking-demand).

**Business Problem:** Canceled bookings result in a loss of revenue for
the hotels and the hotels would like to reduce the number of canceled
bookings.

The first goal of this project was to provide insights about how
different variables affected cancellation status. These insights can be
used by the hotels to guide business strategy. Trends in booking data
were also analyzed to aid with demand forecasting.

The second goal was to use machine learning to develop a tool that can
be used for estimating the number of canceled bookings, in order to
support the hotels when calculating revenue projections.

##### The project was done using R and it consisted of two parts:

- **Part 1:** An Exploratory Data Analysis was performed in order to
  gain insights to support hotel management with operations and demand
  forecasting decisions. An overview of booking results was also
  presented.

- **Part 2:** The Random Forest machine learning algorithm was used to
  predict cancellation status, indicating if a booking was canceled or
  not canceled. Several Random Forest machine learning models were built
  and compared, in order to evaluate which model was to be selected in
  providing estimates to support the hotels when calculating revenue
  projections.

<br>

![A bar chart of the Number of Bookings by Cancellation Status and Hotel](/images/8.png)

- The City Hotel had more canceled bookings than the Resort Hotel.
- City Hotel cancellation rate: **41.73%**. ((33,102)/(46,228 +
  33,102))x100%  
- Resort Hotel cancellation rate: **27.76%**. ((11,122)/(28,938 +
  11,122))x100%
- The City Hotel also had a higher cancellation rate than the Resort
  Hotel. 
  
  <br>
  
  ![A doughnut chart of the Deposit Type Percentage for Bookings](/images/10.png)
  
Most of the bookings for both of the hotels were made with no deposit.

<br>
  
![A bar chart of the Year-Over-Year Bookings Kept in July](/images/14.png)

From 2015 through 2017, the City Hotel had an upward trend in its number
of bookings kept in July.

From 2015 through 2017, the Resort Hotel had a more constant level in
its number of bookings kept in July, with a slight decrease in 2016.

<br>

![A bar chart of the Year-Over-Year Bookings Kept in August](/images/15.png)

From 2015 to 2016, the City Hotel had a **70.75%** increase in its
number of bookings kept in August. ((2,131 - 1,248)/(1,248))x100%  
From 2016 to 2017, the City Hotel had a **6.05%** decrease in its number
of bookings kept in August. ((2,002 - 2,131)/(2,131))x100%

From 2015 to 2016, the Resort Hotel had a slight increase in the number
of bookings kept in August. From 2016 to 2017, the Resort Hotel had no
change in the number of bookings kept in August. 

<br>

Random Forest is an ensemble machine learning algorithm that can be used
for regression or classification purposes. It combines multiple Decision
Trees in order to make a prediction. This project used Random Forest as
a classifier. Random Forest was used to predict if a booking will be canceled or not
canceled.

Several Random Forest machine learning models were built and compared,
with the goal of selecting a model that would generalize well with new
data, have a low number of false negatives, and have a high accuracy.
The selected model can be used by the two hotels to estimate how many
canceled and non-canceled bookings they may have for a given time
period.

<br>

**Conclusion:** The results of the fourth model were very close to the
results of the second model, which used all the features when building a
Random Forest. The second model provided the highest accuracy and the
lowest number of false negatives, but it was not selected because using
every feature may overfit the data and not allow the model to predict
well with new data.  
Since the fourth model used less features and had an accuracy and number
of false negatives that were comparable to the second model, the fourth
model was selected for predicting whether a booking will be canceled or
not canceled. 

<br>


