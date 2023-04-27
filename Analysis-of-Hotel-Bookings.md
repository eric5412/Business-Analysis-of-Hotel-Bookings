Business Analysis of Hotel Bookings
================

<br>

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

#### The project was done using R and it consisted of two parts:

- **Part 1:** An Exploratory Data Analysis was performed in order to
  gain insights to support hotel management with operations and demand
  forecasting decisions. An overview of booking results was also
  presented.

- **Part 2:** The Random Forest machine learning algorithm was used to
  predict cancellation status, indicating if a booking would be canceled or
  not canceled. Four Random Forest machine learning models were built
  and compared, in order to evaluate which model was to be selected in
  providing estimates to support the hotels when calculating revenue
  projections.

<br>

### Loading the Packages:

``` r
library(tidyverse)
library(corrplot)
library(reshape2)
library(scales)
library(caret)
library(randomForest)
```

<br>

### Loading the Data:

``` r
# To load the data and to save it to a data frame called bookings_df:
bookings_df <- read_csv("hotel_bookings.csv")
```

<br>

### Viewing the Data:

``` r
# To view the first six rows of the data:
head(bookings_df)
```

    ## # A tibble: 6 × 32
    ##   hotel   is_ca…¹ lead_…² arriv…³ arriv…⁴ arriv…⁵ arriv…⁶ stays…⁷ stays…⁸ adults
    ##   <chr>     <dbl>   <dbl>   <dbl> <chr>     <dbl>   <dbl>   <dbl>   <dbl>  <dbl>
    ## 1 Resort…       0     342    2015 July         27       1       0       0      2
    ## 2 Resort…       0     737    2015 July         27       1       0       0      2
    ## 3 Resort…       0       7    2015 July         27       1       0       1      1
    ## 4 Resort…       0      13    2015 July         27       1       0       1      1
    ## 5 Resort…       0      14    2015 July         27       1       0       2      2
    ## 6 Resort…       0      14    2015 July         27       1       0       2      2
    ## # … with 22 more variables: children <dbl>, babies <dbl>, meal <chr>,
    ## #   country <chr>, market_segment <chr>, distribution_channel <chr>,
    ## #   is_repeated_guest <dbl>, previous_cancellations <dbl>,
    ## #   previous_bookings_not_canceled <dbl>, reserved_room_type <chr>,
    ## #   assigned_room_type <chr>, booking_changes <dbl>, deposit_type <chr>,
    ## #   agent <chr>, company <chr>, days_in_waiting_list <dbl>,
    ## #   customer_type <chr>, adr <dbl>, required_car_parking_spaces <dbl>, …

The data consists of columns that have numeric, date, and character data
types. A character data type stores character values or strings. For the
analysis, the character data types were converted to factors. A factor
is a data type that is used to store a categorical variable which has a
limited number of different values. These values are called levels. <br>
<br>

The columns that had character data types were converted to factors, so
that they could be interpreted as categorical variables instead of
string values.

``` r
# To convert the character columns to factors:
bookings_df <- as.data.frame(unclass(bookings_df), stringsAsFactors = TRUE)
```

<br>

### Viewing the Structure of the Data:

``` r
glimpse(bookings_df)
```

    ## Rows: 119,390
    ## Columns: 32
    ## $ hotel                          <fct> Resort Hotel, Resort Hotel, Resort Hote…
    ## $ is_canceled                    <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, …
    ## $ lead_time                      <dbl> 342, 737, 7, 13, 14, 14, 0, 9, 85, 75, …
    ## $ arrival_date_year              <dbl> 2015, 2015, 2015, 2015, 2015, 2015, 201…
    ## $ arrival_date_month             <fct> July, July, July, July, July, July, Jul…
    ## $ arrival_date_week_number       <dbl> 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,…
    ## $ arrival_date_day_of_month      <dbl> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, …
    ## $ stays_in_weekend_nights        <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    ## $ stays_in_week_nights           <dbl> 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, …
    ## $ adults                         <dbl> 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, …
    ## $ children                       <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    ## $ babies                         <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    ## $ meal                           <fct> BB, BB, BB, BB, BB, BB, BB, FB, BB, HB,…
    ## $ country                        <fct> PRT, PRT, GBR, GBR, GBR, GBR, PRT, PRT,…
    ## $ market_segment                 <fct> Direct, Direct, Direct, Corporate, Onli…
    ## $ distribution_channel           <fct> Direct, Direct, Direct, Corporate, TA/T…
    ## $ is_repeated_guest              <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    ## $ previous_cancellations         <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    ## $ previous_bookings_not_canceled <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    ## $ reserved_room_type             <fct> C, C, A, A, A, A, C, C, A, D, E, D, D, …
    ## $ assigned_room_type             <fct> C, C, C, A, A, A, C, C, A, D, E, D, E, …
    ## $ booking_changes                <dbl> 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    ## $ deposit_type                   <fct> No Deposit, No Deposit, No Deposit, No …
    ## $ agent                          <fct> NULL, NULL, NULL, 304, 240, 240, NULL, …
    ## $ company                        <fct> NULL, NULL, NULL, NULL, NULL, NULL, NUL…
    ## $ days_in_waiting_list           <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    ## $ customer_type                  <fct> Transient, Transient, Transient, Transi…
    ## $ adr                            <dbl> 0.00, 0.00, 75.00, 75.00, 98.00, 98.00,…
    ## $ required_car_parking_spaces    <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    ## $ total_of_special_requests      <dbl> 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 3, …
    ## $ reservation_status             <fct> Check-Out, Check-Out, Check-Out, Check-…
    ## $ reservation_status_date        <date> 2015-07-01, 2015-07-01, 2015-07-02, 20…

<br>

### Summary Statistics:

``` r
summary(bookings_df)
```

    ##           hotel        is_canceled       lead_time   arrival_date_year
    ##  City Hotel  :79330   Min.   :0.0000   Min.   :  0   Min.   :2015     
    ##  Resort Hotel:40060   1st Qu.:0.0000   1st Qu.: 18   1st Qu.:2016     
    ##                       Median :0.0000   Median : 69   Median :2016     
    ##                       Mean   :0.3704   Mean   :104   Mean   :2016     
    ##                       3rd Qu.:1.0000   3rd Qu.:160   3rd Qu.:2017     
    ##                       Max.   :1.0000   Max.   :737   Max.   :2017     
    ##                                                                       
    ##  arrival_date_month arrival_date_week_number arrival_date_day_of_month
    ##  August :13877      Min.   : 1.00            Min.   : 1.0             
    ##  July   :12661      1st Qu.:16.00            1st Qu.: 8.0             
    ##  May    :11791      Median :28.00            Median :16.0             
    ##  October:11160      Mean   :27.17            Mean   :15.8             
    ##  April  :11089      3rd Qu.:38.00            3rd Qu.:23.0             
    ##  June   :10939      Max.   :53.00            Max.   :31.0             
    ##  (Other):47873                                                        
    ##  stays_in_weekend_nights stays_in_week_nights     adults      
    ##  Min.   : 0.0000         Min.   : 0.0         Min.   : 0.000  
    ##  1st Qu.: 0.0000         1st Qu.: 1.0         1st Qu.: 2.000  
    ##  Median : 1.0000         Median : 2.0         Median : 2.000  
    ##  Mean   : 0.9276         Mean   : 2.5         Mean   : 1.856  
    ##  3rd Qu.: 2.0000         3rd Qu.: 3.0         3rd Qu.: 2.000  
    ##  Max.   :19.0000         Max.   :50.0         Max.   :55.000  
    ##                                                               
    ##     children           babies                 meal          country     
    ##  Min.   : 0.0000   Min.   : 0.000000   BB       :92310   PRT    :48590  
    ##  1st Qu.: 0.0000   1st Qu.: 0.000000   FB       :  798   GBR    :12129  
    ##  Median : 0.0000   Median : 0.000000   HB       :14463   FRA    :10415  
    ##  Mean   : 0.1039   Mean   : 0.007949   SC       :10650   ESP    : 8568  
    ##  3rd Qu.: 0.0000   3rd Qu.: 0.000000   Undefined: 1169   DEU    : 7287  
    ##  Max.   :10.0000   Max.   :10.000000                     ITA    : 3766  
    ##  NA's   :4                                               (Other):28635  
    ##        market_segment  distribution_channel is_repeated_guest
    ##  Online TA    :56477   Corporate: 6677      Min.   :0.00000  
    ##  Offline TA/TO:24219   Direct   :14645      1st Qu.:0.00000  
    ##  Groups       :19811   GDS      :  193      Median :0.00000  
    ##  Direct       :12606   TA/TO    :97870      Mean   :0.03191  
    ##  Corporate    : 5295   Undefined:    5      3rd Qu.:0.00000  
    ##  Complementary:  743                        Max.   :1.00000  
    ##  (Other)      :  239                                         
    ##  previous_cancellations previous_bookings_not_canceled reserved_room_type
    ##  Min.   : 0.00000       Min.   : 0.0000                A      :85994     
    ##  1st Qu.: 0.00000       1st Qu.: 0.0000                D      :19201     
    ##  Median : 0.00000       Median : 0.0000                E      : 6535     
    ##  Mean   : 0.08712       Mean   : 0.1371                F      : 2897     
    ##  3rd Qu.: 0.00000       3rd Qu.: 0.0000                G      : 2094     
    ##  Max.   :26.00000       Max.   :72.0000                B      : 1118     
    ##                                                        (Other): 1551     
    ##  assigned_room_type booking_changes       deposit_type        agent      
    ##  A      :74053      Min.   : 0.0000   No Deposit:104641   9      :31961  
    ##  D      :25322      1st Qu.: 0.0000   Non Refund: 14587   NULL   :16340  
    ##  E      : 7806      Median : 0.0000   Refundable:   162   240    :13922  
    ##  F      : 3751      Mean   : 0.2211                       1      : 7191  
    ##  G      : 2553      3rd Qu.: 0.0000                       14     : 3640  
    ##  C      : 2375      Max.   :21.0000                       7      : 3539  
    ##  (Other): 3530                                            (Other):42797  
    ##     company       days_in_waiting_list         customer_type  
    ##  NULL   :112593   Min.   :  0.000      Contract       : 4076  
    ##  40     :   927   1st Qu.:  0.000      Group          :  577  
    ##  223    :   784   Median :  0.000      Transient      :89613  
    ##  67     :   267   Mean   :  2.321      Transient-Party:25124  
    ##  45     :   250   3rd Qu.:  0.000                             
    ##  153    :   215   Max.   :391.000                             
    ##  (Other):  4354                                               
    ##       adr          required_car_parking_spaces total_of_special_requests
    ##  Min.   :  -6.38   Min.   :0.00000             Min.   :0.0000           
    ##  1st Qu.:  69.29   1st Qu.:0.00000             1st Qu.:0.0000           
    ##  Median :  94.58   Median :0.00000             Median :0.0000           
    ##  Mean   : 101.83   Mean   :0.06252             Mean   :0.5714           
    ##  3rd Qu.: 126.00   3rd Qu.:0.00000             3rd Qu.:1.0000           
    ##  Max.   :5400.00   Max.   :8.00000             Max.   :5.0000           
    ##                                                                         
    ##  reservation_status reservation_status_date
    ##  Canceled :43017    Min.   :2014-10-17     
    ##  Check-Out:75166    1st Qu.:2016-02-01     
    ##  No-Show  : 1207    Median :2016-08-07     
    ##                     Mean   :2016-07-30     
    ##                     3rd Qu.:2017-02-08     
    ##                     Max.   :2017-09-14     
    ## 

##### **Findings:**

- The arrival date years were 2015, 2016, and 2017.
- There were four missing values in the overall data frame.
- There was a negative value in the “adr” column. The “adr” column is
  the Average Daily Rate and is provided in Euro (€).
- For a factor with more than six categories, the summary grouped the
  remaining counts of rows in an (Other) category. <br> <br>

## **Part 1:**

### Exploratory Data Analysis:

``` r
# Box plot of Lead Time by Cancellation Status:
ggplot(bookings_df, aes(x = factor(is_canceled), y = lead_time)) + 
  geom_boxplot() + 
  labs(title = "Lead Time by Cancellation Status",
     subtitle = "0 = Booking was not canceled,  1 = Booking was canceled",
     x="Cancellation Status",
     y="Lead Time") +
  theme(text = element_text(size = 12.25))
```

![A box plot of Lead Time by Cancellation Status](/images/1.png)

The Lead Time is the number of days that have elapsed since a booking
was made. The box plot suggests that as Lead Time increases, the
likelihood of a booking being canceled also tends to increase. <br> <br>

``` r
# To view the values of the box plot:
bookings_df %>%                               
  group_by(is_canceled) %>%
  summarize(min = min(lead_time, 
                      na.rm=TRUE),
            q1 = quantile(lead_time, 
                          0.25, na.rm=TRUE),
            median = median(lead_time, 
                            na.rm=TRUE),
            mean = mean(lead_time, 
                        na.rm=TRUE),
            q3 = quantile(lead_time, 
                          0.75, na.rm=TRUE),
            max = max(lead_time, 
                      na.rm=TRUE)) %>% ungroup()
```

    ## # A tibble: 2 × 7
    ##   is_canceled   min    q1 median  mean    q3   max
    ##         <dbl> <dbl> <dbl>  <dbl> <dbl> <dbl> <dbl>
    ## 1           0     0     9     45  80.0   124   737
    ## 2           1     0    48    113 145.    214   629

In general, the Lead Time was higher for bookings that were canceled.
The numerical summary highlights that the median Lead Time for bookings
that were canceled was 113 days, compared to 45 days for the bookings
that were not canceled. The mean Lead Time was also higher for bookings
that were canceled. There were two high values for bookings that were
not canceled, but these appear to be outliers. The third quartile Lead
Time for canceled bookings was also higher at 214 days compared to 124
days for bookings that were not canceled.

The numerical summary suggests that if a booking is made further in
advance, it may be more prone to a cancellation. A possible strategy to
reduce cancellations could be to send periodic emails to potential
guests that have booked a hotel far in advance, with information about
their booking and any promotions the hotel may have. <br> <br>

``` r
# Scatter plot of Previous Cancellations by Cancellation Status:
ggplot(bookings_df, aes(x = is_canceled, y = previous_cancellations)) +
  geom_point() +
  labs(x = "Cancellation Status",
       y = "Previous Cancellations",
       title = "Previous Cancellations by Cancellation Status") +
  theme(text = element_text(size = 12.25))
```

![A scatter plot of Previous Cancellations by Cancellation Status](/images/2.png)

The scatter plot suggests that when there is an increase in the number
of Previous Cancellations by a potential guest, the likelihood of their
current booking being canceled also increases. <br> <br>

``` r
# Bar chart of the Number of Bookings by Total Special Requests and Cancellation 
# Status:
bookings_df %>%
  ggplot(aes(x = factor(total_of_special_requests))) +
  geom_bar(width = 0.3) +
  geom_text(aes(label = comma(after_stat(count))), stat = "count", vjust = -0.5,
            color = "black") +
  labs(title = "Number of Bookings by Total Special Requests 
and Cancellation Status",
       x="Total Special Requests",
       y="Number of Bookings") +
  theme(text = element_text(size = 12.25)) +
  facet_wrap(~is_canceled) +
  scale_y_continuous(labels = label_comma(), limits = c(0, 38000))
```

![A bar chart of the Number of Bookings by Total Special Requests and Cancellation Status](/images/3.png)

The bar chart shows that for each category of Total Special Requests,
there were more bookings that were not canceled, in comparison to
bookings that were canceled. This suggests that when a booking has
several special requests included, the likelihood of the booking being
canceled decreases. A possible strategy to decrease cancellations may be
to prioritize asking the potential guest if they have any special
requests for their booking, such as the size of bed needed or if they
prefer their room to be on the first floor. <br> <br>

``` r
# Bar chart of the Number of Bookings by Required Car Parking Spaces and 
# Cancellation Status:
bookings_df %>%
  ggplot(aes(x = factor(required_car_parking_spaces))) +
  geom_bar(width = 0.3) +
  geom_text(aes(label = comma(after_stat(count))), stat = "count", vjust = -0.5,
            color = "black") +
  labs(title = "Number of Bookings by Required Car Parking Spaces 
and Cancellation Status",
       x="Required Car Parking Spaces",
       y="Number of Bookings") +
  theme(text = element_text(size = 12.25)) +
  facet_wrap(~is_canceled) +
  scale_y_continuous(labels = label_comma(), limits = c(0, 70000))
```

![A bar chart of the Number of Bookings by Required Car Parking Spaces and Cancellation Status](/images/4.png)

The bar chart suggests that if the option to request car parking spaces
is used during a booking, then there may be a lower likelihood that the
booking will be canceled. <br> <br>

``` r
# Bar chart of the Number of Bookings by Booking Changes and
# Cancellation Status:
bookings_df %>%
  ggplot(aes(x = factor(booking_changes))) +
  geom_bar(width = 0.3) +
  geom_text(aes(label = comma(after_stat(count))), stat = "count", vjust = -0.5,
            color = "black", size = 2.3, hjust = 0.30) +
  labs(title = "Number of Bookings by Booking Changes
and Cancellation Status",
       x="Booking Changes",
       y="Number of Bookings") +
  theme(text = element_text(size = 12.25)) +
  facet_wrap(~is_canceled) +
  scale_y_continuous(labels = label_comma(), limits = c(0, 62000))
```

![A bar chart of the Number of Bookings by Booking Changes and Cancellation Status](/images/5.png)

The bar chart suggests that when several changes have been made to a
booking, there is a lower likelihood that the booking will be canceled.
<br> <br>

``` r
# Bar chart of the Number of Bookings by Repeated Guest Status and 
# Cancellation Status:
bookings_df %>%
  ggplot(aes(x = factor(is_repeated_guest))) +
  geom_bar(width = 0.3) +
  geom_text(aes(label = comma(after_stat(count))), stat = "count", vjust = -0.5,
            color = "black") +
  labs(title = "Number of Bookings by Repeated Guest Status and 
Cancellation Status",
       x="Repeated Guest Status",
       y="Number of Bookings") +
  theme(text = element_text(size = 12.25)) +
  facet_wrap(~is_canceled) +
  scale_y_continuous(labels = label_comma(), limits = c(0, 74000))
```

![A bar chart of the Number of Bookings by Repeated Guest Status and Cancellation Status](/images/6.png)

The bar chart suggests that if a potential guest is a repeated guest,
then there may be a lower likelihood that their current booking will be
canceled.

- Repeated Guest cancellation rate: **14.49%**. ((552)/(3,258 +
  552))x100%  
- Repeated Guest non-cancellation rate: **85.51%**. ((3,258)/(3,258 +
  552))x100%  
- Most of the repeated guests did not cancel their current booking. <br>
  <br>

### Business Overview:

The number of bookings by cancellation status from 2015 to 2017 was
analyzed.

``` r
# To view the Number of Bookings by Cancellation Status:
bookings_df %>%
  ggplot(aes(x = factor(is_canceled))) +
  geom_bar(width = 0.3) +
  geom_text(aes(label = comma(after_stat(count))), stat = "count", vjust = -0.5,
            color = "black") +
  labs(title = "Number of Bookings by Cancellation Status",
       subtitle = "0 = Booking was not canceled,  1 = Booking was canceled",
       x="Cancellation Status",
       y="Number of Bookings") +
  theme(text = element_text(size = 12.25)) +
  scale_y_continuous(labels = label_comma(), limits = c(0, 81000))
```

![A bar chart of the Number of Bookings by Cancellation Status](/images/7.png)

Overall, most of the bookings for the hotels were not canceled during
2015 to 2017. <br> <br>

The number of bookings by cancellation status during 2015 to 2017 was
then compared between the City Hotel and the Resort Hotel.

``` r
# To view the Number of Bookings by Cancellation Status and Hotel:
bookings_df %>% count(is_canceled, hotel) %>%
  ggplot(aes(x = factor(is_canceled), y = n)) +
  geom_col(aes(fill = hotel), position = position_dodge(0.6), width = 0.5) +
  geom_text(aes(label = comma(n), group = hotel), 
            position = position_dodge(0.6),
            vjust = -0.4, size = 4.0) +
  labs(title = "Number of Bookings by Cancellation Status and Hotel",
       x="Cancellation Status",
       y="Number of Bookings",
       fill = "Hotel",
       caption = "From 2015 through 2017") +
  theme(text = element_text(size = 12.25)) +
  scale_y_continuous(labels = label_comma(), limits = c(0, 50000))
```

![A bar chart of the Number of Bookings by Cancellation Status and Hotel](/images/8.png)

- A cancellation status of 0 indicates that a booking was not canceled. A cancellation status of 1 indicates that a booking was canceled.
- The City Hotel had more canceled bookings than the Resort Hotel.
- City Hotel cancellation rate: **41.73%**. ((33,102)/(46,228 +
  33,102))x100%  
- Resort Hotel cancellation rate: **27.76%**. ((11,122)/(28,938 +
  11,122))x100%
- The City Hotel also had a higher cancellation rate than the Resort
  Hotel. <br> <br>

The distribution of the number of bookings by month, year, and hotel was
analyzed to determine the completeness of the booking data.

``` r
# To view the distribution of the Number of Bookings by Month, Year, and Hotel:
bookings_df %>%
  ggplot(aes(x = factor(arrival_date_month, 
                        levels = c("January", "February", "March", "April", 
                                   "May","June", "July", "August", "September",
                                   "October", "November", "December")))) +
  geom_bar() +
  facet_grid(~arrival_date_year~hotel) +
  labs(title = "Distribution of the Number of Bookings by Month, Year, and Hotel",
       x="Month",
       y="Number of Bookings") +
  theme(axis.text.x = element_text(angle = 45))
```

![A bar chart of the Number of Bookings by Month, Year, and Hotel](/images/9.png)

Booking data was only provided for the last six months in 2015. Booking
data was provided for all twelve months in 2016. Booking data was only
provided for the first eight months in 2017.

July and August were the two months where data was provided for all
three years. These months will be used for Year-Over-Year analysis. <br>
<br>

A doughnut chart was used to analyze the Deposit Type Percentage.

``` r
# Doughnut chart to analyze the Deposit Type Percentage for bookings:
bookings_df %>% count(deposit_type) %>% 
  mutate(prop = n/sum(n), percent = round((prop*100), 2)) %>%
  ggplot(aes(x = 3, y = percent, fill = deposit_type)) +
  geom_col() +
  geom_text(aes(label = paste0(percent, "%")),
            position = position_stack(vjust = 0.5)) +
  coord_polar(theta = "y") +
  xlim(c(0.2, 3 + 0.5)) +
  labs(x = NULL, y = NULL, fill = "Deposit Type") +
  theme_classic() +
  theme(axis.line = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank()) +
  labs(title = "Deposit Type Percentage",
       fill = "Deposit Type",
       caption = "From 2015 through 2017") +
  theme(text = element_text(size = 12.75))
```

![A doughnut chart of the Deposit Type Percentage for Bookings](/images/10.png)

Most of the bookings for both of the hotels were made with no deposit.
<br> <br>

To compare the number of bookings that were kept by month and hotel in
2016, the data was filtered by the “is_canceled” column and the
“arrival_date_year” column. A line chart was created for this
comparison.

``` r
# To filter the data to only show bookings that were not canceled, in order to
# gain insights about the bookings that were kept:
not_canceled_df <- bookings_df %>%
  filter(is_canceled == 0)

# Line chart of the Number of Bookings Kept by Month and Hotel in 2016:
not_canceled_df %>% filter(arrival_date_year == 2016) %>%
  count(arrival_date_year, arrival_date_month, hotel) %>%
  ggplot(aes(x = factor(arrival_date_month,
                        levels = c("January", "February", "March", "April",
                                   "May", "June", "July", "August", "September",
                                   "October", "November", "December")), 
             y = n, color = hotel, group = hotel)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(title = "Number of Bookings Kept by Month and Hotel in 2016",
       x ="Month",
       y ="Number of Bookings Kept",
       color = "Hotel") +
  geom_text(aes(label = comma(n),
                vjust = -0.8, hjust = 0.8), show.legend = FALSE) +
  theme(text = element_text(size = 12.25)) +
  scale_y_continuous(labels = label_comma(), limits = c(750, 2500)) +
  theme(axis.text.x = element_text(angle = 25))
```

![A line chart of the Number of Bookings Kept by Month and Hotel in 2016](/images/11.png)

For every month of 2016, the City Hotel received more bookings kept than
the Resort Hotel. In September, the City Hotel received its highest
number of bookings kept. In October, the Resort Hotel received its
highest number of bookings kept. In January, both the City Hotel and the
Resort Hotel received their lowest number of bookings kept.

Based on the data from 2016, the City Hotel can expect its highest
demand in September and its lowest demand in January. The Resort Hotel
can expect its highest demand in October and its lowest demand in
January. <br> <br>

A box plot was used to compare the Average Daily Rate by hotel for the
bookings that were kept during 2015 to 2017.

``` r
# Box plot of Average Daily Rate by Hotel for Bookings Kept:
ggplot(not_canceled_df, aes(x = factor(hotel), y = adr)) + 
  geom_boxplot() + 
  labs(title = "Average Daily Rate by Hotel for Bookings Kept",
       x ="Hotel",
       y ="Average Daily Rate",
       caption = "From 2015 through 2017") +
  theme(text = element_text(size = 12.25)) +
  scale_y_continuous(labels = label_dollar(prefix = "\u20ac", suffix = ""))
```

![A box plot of the Average Daily Rate by Hotel for Bookings Kept](/images/12.png)

On average, the City Hotel had a higher Average Daily Rate. <br> <br>

``` r
# To view the values of the box plot:
not_canceled_df %>%                               
  group_by(hotel) %>%
  summarize(min = min(adr, 
                      na.rm=TRUE),
            q1 = quantile(adr, 
                          0.25, na.rm=TRUE),
            median = median(adr, 
                            na.rm=TRUE),
            mean = mean(adr, 
                        na.rm=TRUE),
            q3 = quantile(adr, 
                          0.75, na.rm=TRUE),
            max = max(adr, 
                      na.rm=TRUE)) %>% ungroup()
```

    ## # A tibble: 2 × 7
    ##   hotel          min    q1 median  mean    q3   max
    ##   <fct>        <dbl> <dbl>  <dbl> <dbl> <dbl> <dbl>
    ## 1 City Hotel    0       80   99.9 106.   126    510
    ## 2 Resort Hotel -6.38    48   72    90.8  118.   508

The City Hotel had two distinct outlier values for the Average Daily
Rate. For one of these bookings, one guest only stayed one night and the
Average Daily Rate listed for their booking was €510.00. The Resort
Hotel had two similar distinct outlier values. For one of these
bookings, two guests stayed one night and the Average Daily Rate listed
for their booking was €508.00.

The Resort Hotel had a booking where the Average Daily Rate was a
negative €6.38. The Resort Hotel may want to see if this was due to a
data entry error. <br> <br>

For the City Hotel, there were several bookings that were kept that were
listed as having a €0 Average Daily Rate.

``` r
# Number of Bookings Kept at the City Hotel with a €0 Average Daily Rate:
not_canceled_df %>% filter(hotel == "City Hotel" & adr == 0) %>% count(adr)
```

    ##   adr    n
    ## 1   0 1079

The City Hotel had 1,079 bookings where the Average Daily Rate for the
booking was €0. Some of these bookings showed that a guest stayed
several days. The City Hotel may want to see if this was a data entry
error. <br> <br>

For the Resort Hotel, there were also several bookings that were kept
that were listed as having a €0 Average Daily Rate.

``` r
# Number of Bookings Kept at the Resort Hotel with a €0 Average Daily Rate:
not_canceled_df %>% filter(hotel == "Resort Hotel" & adr == 0) %>% count(adr)
```

    ##   adr   n
    ## 1   0 667

The Resort Hotel had 667 bookings where the Average Daily Rate for the
booking was €0. Some of these bookings showed that a guest stayed
several days. The Resort Hotel may want to see if this was a data entry
error. <br> <br>

A new column was created to calculate the total nights stayed by hotel
during 2016.

``` r
# To create a new column that calculates the Total Stay for a booking, by adding
# the "stays_in_weekend_nights" and the "stays_in_week_nights" columns:
not_canceled_df <- not_canceled_df %>% 
  mutate(total_stay = stays_in_weekend_nights + stays_in_week_nights)

# Bar chart of the Total Nights Stayed by Hotel in 2016:
not_canceled_df %>% filter(arrival_date_year == "2016") %>%
  group_by(hotel) %>%
  summarize(sum = sum(total_stay, na.rm=TRUE)) %>% ungroup() %>%
  ggplot(aes(x = factor(hotel), y = sum)) +
  geom_col(aes(fill = hotel), position = position_dodge(0.6), width = 0.3) +
  geom_text(
    aes(label = comma(sum), group = hotel),
    position = position_dodge(0.6),
    vjust = -0.5, size = 4.0) +
  labs(title = "Total Nights Stayed by Hotel in 2016",
       x="Hotel",
       y="Total Nights Stayed",
       fill = "Hotel") +
  scale_y_continuous(labels = comma, limits = c(0, 67000)) +
  theme(text = element_text(size = 12.25)) +
  theme(legend.position = "none")
```

![A bar chart of the Total Nights Stayed by Hotel in 2016](/images/13.png)

For 2016, there were more nights stayed in the City Hotel than the
Resort Hotel. <br> <br>

The Year-Over-Year number of bookings kept in July was compared for the
City Hotel and the Resort Hotel.

``` r
# Bar chart of Year-Over-Year Bookings Kept in July:
not_canceled_df %>% filter(arrival_date_month == "July") %>%
  ggplot(aes(x = factor(arrival_date_year))) +
  geom_bar(width = 0.5) +
  geom_text(aes(label = comma(after_stat(count))), stat = "count", 
            vjust = -0.25, color = "black") +
  labs(title = "Year-Over-Year Bookings Kept in July",
       x="Year",
       y="Bookings Kept") +
  facet_wrap(~hotel) +
  theme(text = element_text(size = 12.25)) +
  scale_y_continuous(labels = label_comma(), limits = c(0, 2500))
```

![A bar chart of the Year-Over-Year Bookings Kept in July](/images/14.png)

From 2015 through 2017, the City Hotel had an upward trend in its number
of bookings kept in July.

From 2015 through 2017, the Resort Hotel had a more constant level in
its number of bookings kept in July, with a slight decrease in 2016.
<br> <br>

The Year-Over-Year number of bookings kept in August was compared for
the City Hotel and the Resort Hotel.

``` r
# Bar chart of Year-Over-Year Bookings Kept in August:
not_canceled_df %>% filter(arrival_date_month == "August") %>%
  ggplot(aes(x = factor(arrival_date_year))) +
  geom_bar(width = 0.5) +
  geom_text(aes(label = comma(after_stat(count))), stat = "count", 
            vjust = -0.25, color = "black") +
  labs(title = "Year-Over-Year Bookings Kept in August",
       x="Year",
       y="Bookings Kept") +
  facet_wrap(~hotel) +
  theme(text = element_text(size = 12.25)) +
  scale_y_continuous(labels = label_comma(), limits = c(0, 2500))
```

![A bar chart of the Year-Over-Year Bookings Kept in August](/images/15.png)

From 2015 to 2016, the City Hotel had a **70.75%** increase in its
number of bookings kept in August. ((2,131 - 1,248)/(1,248))x100%  
From 2016 to 2017, the City Hotel had a **6.05%** decrease in its number
of bookings kept in August. ((2,002 - 2,131)/(2,131))x100%

From 2015 to 2016, the Resort Hotel had a slight increase in the number
of bookings kept in August. From 2016 to 2017, the Resort Hotel had no
change in the number of bookings kept in August. <br> <br>

## **Part 2:**

### Random Forest Machine Learning Models:

Random Forest is an ensemble machine learning algorithm that can be used
for regression or classification purposes. It combines multiple Decision
Trees in order to make a prediction. This project used Random Forest as
a classifier.

Four Random Forest machine learning models were built and compared,
with the goal of selecting a model that would generalize well with new
data, have a low number of false negatives, and have a high accuracy.
The selected model can be used by the two hotels to estimate how many
canceled and non-canceled bookings they may have for a given time
period.

Random Forest was used to predict if a booking would be canceled or not
canceled based on certain independent variables, which are also called
features. The “is_canceled” column is the dependent variable, which is
also called the target variable.

##### For the “is_canceled” column:

- 0 means that a booking was not canceled, which can be interpreted as
  being negative for a cancellation.
- 1 means that a booking was canceled, which can be interpreted as being
  positive for a cancellation. <br> <br>

The data frame was copied to a new data frame that was used for building
the Random Forest machine learning models.

``` r
# To copy the original data frame to a new data frame that will be used for
# building Random Forest machine learning models: 
bookings_ml <- bookings_df
```

<br>

The “is_canceled” and “is_repeated_guest” binary columns are numeric
data types. They needed to be converted to factors, so that they could
be interpreted as categorical variables when using Random Forest.
Several Random Forest machine learning models were built and compared,
to determine which model was to be selected in providing estimates to
support the hotels when calculating revenue projections.

``` r
# To convert the "is_canceled" and "is_repeated_guest" binary columns to 
# factors in order to use them as categorical variables in the Random Forest
# machine learning model:
bookings_ml$is_canceled <- as.factor(bookings_ml$is_canceled)
bookings_ml$is_repeated_guest <- as.factor(bookings_ml$is_repeated_guest)
```

``` r
# To view how the model interprets the "is_canceled" target variable:
contrasts(bookings_ml$is_canceled)
```

    ##   1
    ## 0 0
    ## 1 1

0 = the booking was not canceled  
1 = the booking was canceled <br> <br>

The rows that contained at least one missing value were removed.

``` r
# To remove the rows that contain at least one missing value:
bookings_ml_v1 <- na.omit(bookings_ml)
```

The rows that had at least one missing value were removed instead of
imputing the missing values with the mean or the median, because there
were very few rows with missing values in relation to the total number
of rows in the data frame. A total of four rows were removed out of
119,390 rows. <br> <br>

When building a Random Forest machine learning model in R, one
consideration is that a factor cannot have more than 53 categories.

``` r
# To determine which factors contain more than 53 categories:
sapply(bookings_ml_v1, function(x) length(unique(x)))
```

The factors with more than 53 categories were “country”, “agent”, and
“company”. Additionally after reviewing the data, these three factors
did not appear to have any relationship with the “is_canceled” target
variable, so they were removed from the data frame. <br> <br>

``` r
# To remove the "country", "agent", and "company" features:
bookings_ml_v2 <- bookings_ml_v1 %>% select(-country, -agent, -company)
```

``` r
# To remove the "reservation_status_date" feature and the reservation_status" 
# feature:
bookings_ml_v3 <- bookings_ml_v2 %>% 
  select(-reservation_status_date, -reservation_status)
```

The “reservation_status_date” feature was removed, because it did not
have any relationship with the “is_canceled” target variable. The
“reservation_status” feature was also removed, because it was not a
predictor, as it only listed the reservation status result of the
current booking. The original data frame contained 31 features. After
removing the five features that were not needed, the data frame now
contained 26 features. <br> <br>

An initial Random Forest model was built. This model only used the
features that had the highest correlations with the target variable, as
determined by the Exploratory Data Analysis.

``` r
# To make the Random Forest model results reproducible:
set.seed(42)
```

``` r
# An initial Random Forest model using only the features that had the highest correlations with the target variable:
rf_model1 <- randomForest(is_canceled ~ total_of_special_requests + 
                          required_car_parking_spaces + booking_changes +
                          previous_cancellations + lead_time + 
                          is_repeated_guest, 
                          data = bookings_ml_v3)
```

``` r
# To display the summary of the initial Random Forest model:
rf_model1
```

    ## 
    ## Call:
    ##  randomForest(formula = is_canceled ~ total_of_special_requests + required_car_parking_spaces + booking_changes + previous_cancellations  + lead_time + is_repeated_guest, data = bookings_ml_v3) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 25.28%
    ## Confusion matrix:
    ##       0     1 class.error
    ## 0 59914 15252   0.2029109
    ## 1 14926 29294   0.3375396

The number of Decision Trees used to build this Random Forest was the
default value of 500 trees. Two variables were tried at each split.

The proportion of Out-Of-Bag samples that are incorrectly classified is
called the Out-Of-Bag error rate. The Out-Of-Bag error rate for the
model was 25.28% and the model accuracy was 74.72%. (100% - 25.28%)

The Out-Of-Bag error rate was calculated as follows:  
(14,926 + 15,252) / (14,926 + 15,252 + 59,914 + 29,294)  
(30,178) / (119,386) = 0.25277 x 100% = 25.277% = 25.28% <br> <br>

The summary of the model also provided a Confusion Matrix. A Confusion
Matrix is used to evaluate the predictions made by a classification
model. It lists the number of true negatives, true positives, false
negatives, and false positives.

0 means that a booking was not canceled, which can be interpreted as
being negative for a cancellation.

1 means that a booking was canceled, which can be interpreted as being
positive for a cancellation.

In the Confusion Matrix that was provided in the summary, the columns
are the target variable values that the model predicted, based on the
features. The rows are the actual values of the target variable from the
data frame that was used to build the model.

For the initial Random Forest model, the number of true negatives was
59,914. The number of true positives was 29,294. The number of false
negatives was 14,926. The number of false positives was 15,252.

In the context of predicting whether a booking will be canceled or not,
an important value to consider is the number of false negatives. In this
situation, a false negative occurs when a booking is predicted to not be
canceled, and it actually ends up being canceled. False negatives may
have more of an effect on revenue projections than false positives.
False negatives may be more costly, because they can result in a lower
actual revenue than what a hotel was expecting in their forecast. <br>
<br>

A second Random Forest model was built to determine if the model
accuracy could be improved. This model used all the features.

``` r
# A second Random Forest model using all the features:
set.seed(42)
rf_model2 <- randomForest(is_canceled ~ ., data = bookings_ml_v3)
```

``` r
# To display the summary of the second Random Forest model:
rf_model2
```

    ## 
    ## Call:
    ##  randomForest(formula = is_canceled ~ ., data = bookings_ml_v3) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 5
    ## 
    ##         OOB estimate of  error rate: 12.61%
    ## Confusion matrix:
    ##       0     1 class.error
    ## 0 70154  5012  0.06667908
    ## 1 10044 34176  0.22713704

The number of Decision Trees used to build this Random Forest was the
default value of 500 trees. Five variables were tried at each split.

With this model, the Out-Of-Bag error rate decreased to 12.61% and the
model accuracy increased to 87.39%. (100% - 12.61%)  
The number of false negatives decreased to 10,044.  
The accuracy improved, but it may not be recommended to use all the
features in the data frame that were used to build this model. If all
the features were used this may possibly overfit the model, and the
model may not generalize well with new data when making predictions.
Some of the features were too specific to the historical data in the
data frame, such as “arrival_date_year” and “arrival_date_week_number”.
These features were not used when building the following models. <br>
<br>

A Variable Importance Plot was created for the second Random Forest
model to determine which features were the most important.

``` r
# To create a Variable Importance Plot that displays the importance of each 
# feature in the model:
varImpPlot(rf_model2) 
```

![A Variable Importance Plot that displays the importance of each feature in the model](/images/16.png)

The Variable Importance Plot measures how important a feature is in
classifying the data. <br> <br>

A third Random Forest model was built. This model used the features from
the first model, as well as the most important features that were
determined by the Variable Importance Plot.

``` r
# A third Random Forest model using the features from the first model and the additional features from the Variable Importance Plot:
set.seed(42)
rf_model3 <- randomForest(is_canceled ~ total_of_special_requests + adr + 
                          required_car_parking_spaces + booking_changes +
                          previous_cancellations + lead_time + deposit_type +
                          is_repeated_guest + arrival_date_day_of_month +
                          market_segment + arrival_date_month + customer_type, 
                          data = bookings_ml_v3)
```

``` r
# To display the summary of the third model:
rf_model3
```

    ## 
    ## Call:
    ##  randomForest(formula = is_canceled ~ total_of_special_requests + adr + required_car_parking_spaces + booking_changes + previous_cancellations + lead_time + deposit_type + is_repeated_guest + arrival_date_day_of_month + market_segment + arrival_date_month + customer_type, data = bookings_ml_v3) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 3
    ## 
    ##         OOB estimate of  error rate: 16.36%
    ## Confusion matrix:
    ##       0     1 class.error
    ## 0 70356  4810   0.0639917
    ## 1 14720 29500   0.3328810

The number of Decision Trees used to build this Random Forest was 500
trees. Three variables were tried at each split.

The Out-Of-Bag error rate for the model increased to 16.36% and the
model accuracy decreased to 83.64%. (100% - 16.36%)  
The number of false negatives increased to 14,720. <br> <br>

The number of trees parameter, ntree, was then evaluated.

``` r
# To evaluate the number of trees parameter:
plot(rf_model3)
```

![A plot of the number of trees parameter, leveling out at 500 trees](/images/17.png)

The plot suggests that the error rate levels out at 500 trees. <br> <br>

The mtry parameter for the third model was then tuned. The mtry
parameter determines how many features to randomly select when splitting
the nodes of the Decision Trees, when building a Random Forest.

``` r
# To tune the mtry parameter which determines how many features to randomly 
# select when splitting the nodes of the Decision Trees:
mtry_tune <- tuneRF(x = bookings_ml_v3%>%select(-is_canceled),
                    y = bookings_ml_v3$is_canceled,mtryStart=2,
                    ntreeTry = 500)
```

    ## mtry = 2  OOB error = 16.26% 
    ## Searching left ...
    ## mtry = 1     OOB error = 23.9% 
    ## -0.4700458 0.05 
    ## Searching right ...
    ## mtry = 4     OOB error = 13.09% 
    ## 0.1949209 0.05 
    ## mtry = 8     OOB error = 12.34% 
    ## 0.05745729 0.05 
    ## mtry = 16    OOB error = 12.54% 
    ## -0.01608852 0.05

![A plot of the optimal number of features to randomly select when splitting the nodes](/images/18.png)

The tuning results suggest that randomly selecting eight variables to
split the nodes of the Decision Trees will result in a lower Out-Of-Bag
error rate. <br> <br>

A fourth Random Forest model was built. This model used the features
from the third model, as well as the tuned mtry parameter:

``` r
# A fourth Random Forest model using the features from the third model and the tuned mtry parameter:
set.seed(42)
rf_model4 <- randomForest(is_canceled ~ total_of_special_requests + adr + 
                          required_car_parking_spaces + booking_changes +
                          previous_cancellations + lead_time + deposit_type +
                          is_repeated_guest + arrival_date_day_of_month +
                          market_segment + arrival_date_month + customer_type, 
                          data = bookings_ml_v3, ntree = 500, mtry = 8)
```

``` r
# To display the summary of the fourth model:
rf_model4
```

    ## 
    ## Call:
    ##  randomForest(formula = is_canceled ~ total_of_special_requests + adr + required_car_parking_spaces + booking_changes + previous_cancellations + lead_time + deposit_type + is_repeated_guest + arrival_date_day_of_month + market_segment + arrival_date_month + customer_type, data = bookings_ml_v3, ntree = 500, mtry = 8) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 8
    ## 
    ##         OOB estimate of  error rate: 13.95%
    ## Confusion matrix:
    ##       0     1 class.error
    ## 0 68766  6400  0.08514488
    ## 1 10250 33970  0.23179557

The number of Decision Trees used to build this Random Forest was 500
trees. Eight variables were tried at each split.

The Out-Of-Bag error rate for the model decreased to 13.95% and the
model accuracy increased to 86.05%. (100% - 13.95%)  
The number of false negatives decreased to 10,250.

**Conclusion:** The results of the fourth model were very close to the
results of the second model, which used all the features when building a
Random Forest. The second model provided the highest accuracy and the
lowest number of false negatives, but it was not selected because using
every feature may overfit the data and not allow the model to predict
well with new data.  
Since the fourth model used less features and had an accuracy and number
of false negatives that were comparable to the second model, the fourth
model was selected for predicting whether a booking will be canceled or
not canceled. <br> <br>

### Using the Selected Model to Make Predictions with New Data:

New data was created to use in the fourth model.

``` r
# To create new data to use with the fourth model:
new_data1 <- data.frame(total_of_special_requests=3,
                       adr=82.00,
                       required_car_parking_spaces=2,
                       booking_changes=2,
                       previous_cancellations=1,
                       lead_time=20,
                       deposit_type=factor("No Deposit", 
                           levels=levels(bookings_ml_v3$deposit_type)),
                       is_repeated_guest=factor("1", 
                           levels=levels(bookings_ml_v3$is_repeated_guest)),
                       arrival_date_day_of_month=1,
                       market_segment=factor("Online TA", 
                           levels=levels(bookings_ml_v3$market_segment)),
                       arrival_date_month=factor("July",
                           levels=levels(bookings_ml_v3$arrival_date_month)),
                       customer_type=factor("Transient", 
                           levels=levels(bookings_ml_v3$customer_type))) 
```

<br>

The fourth model was used to predict if the booking will be canceled or
not canceled.

``` r
# To use the new data with the fourth model to predict if a booking will be 
# canceled or not canceled:
y_prediction1 <- predict(rf_model4, newdata = new_data1)
y_prediction1
```

    ## 1 
    ## 0 
    ## Levels: 0 1

The output was 0, indicating that the model predicted that the booking
would not be canceled. This result supports the observations from the
Exploratory Data Analysis. If a booking has a low lead time, a low
number of previous cancellations, some special requests made, car
parking spaces requested, changes to the booking, and if the potential
guest is a repeated guest, there is a high likelihood that the booking
will not be canceled. <br> <br>

More new data was created to use in the fourth model.

``` r
# To create more new data to use with the fourth model:
new_data2 <- data.frame(total_of_special_requests=0,
                       adr=170.00,
                       required_car_parking_spaces=0,
                       booking_changes=1,
                       previous_cancellations=6,
                       lead_time=180,
                       deposit_type=factor("No Deposit", 
                           levels=levels(bookings_ml_v3$deposit_type)),
                       is_repeated_guest=factor("0", 
                           levels=levels(bookings_ml_v3$is_repeated_guest)),
                       arrival_date_day_of_month=1,
                       market_segment=factor("Online TA", 
                           levels=levels(bookings_ml_v3$market_segment)),
                       arrival_date_month=factor("July",
                           levels=levels(bookings_ml_v3$arrival_date_month)),
                       customer_type=factor("Transient", 
                           levels=levels(bookings_ml_v3$customer_type))) 
```

<br>

The fourth model was used to predict if the booking will be canceled or
not canceled.

``` r
# To use the new data with the fourth model to predict if a booking will be 
# canceled or not canceled:
y_prediction2 <- predict(rf_model4, newdata = new_data2)
y_prediction2
```

    ## 1 
    ## 1 
    ## Levels: 0 1

The output was 1, indicating that the model predicted that the booking
would be canceled. This result supports the observations from the
Exploratory Data Analysis. If a booking has a high lead time, several
previous cancellations, no special requests made, no car parking spaces
requested, few or no booking changes, and if the potential guest is not
a repeated guest, there is a high likelihood that the booking will be
canceled. <br> <br>

### Key Takeaways:

- Large lead times for a booking tend to increase the likelihood that
  the booking will be canceled.

- As the number of previous bookings canceled by a potential guest
  increases, the likelihood that their current booking will be canceled
  also increases.

- When a booking has several special requests included, the likelihood
  of the booking being canceled decreases. A possible strategy to
  decrease cancellations may be to prioritize asking the potential guest
  if they have any special requests for their booking, such as the size
  of bed needed or if they prefer their room to be on the first floor.

- If the option to request car parking spaces is used during a booking,
  then there may be a lower likelihood that the booking will be
  canceled.

- When a potential guest has made several booking changes, there tends
  to be a lower likelihood that their booking will be canceled.

- If a potential guest is a repeated guest, then there may be a lower
  likelihood that their current booking will be canceled.

- Four Random Forest models were built and evaluated. The model accuracy
  and number of false negatives were compared. The fourth model was
  selected for predicting if a booking will be canceled or not canceled.
  <br> <br>

### Recommendations:

- The selected model can be integrated into an app that the hotels can
  use when calculating revenue projections.

- The hotels can periodically run the model on their booking data to
  generate estimates of how many canceled bookings they may expect in a
  given time period.

- The hotels can then take steps that seek to minimize the likelihood of
  canceled bookings, such as:

  - Sending periodic emails to potential guests that have booked a hotel
    far in advance, with information about their booking and any
    promotions the hotel may have.

  - Prioritizing to ask the potential guest if they want to reserve car
    parking spaces and if they have any special requests for their
    booking, such as the size of bed needed or if they prefer their room
    to be on the first floor. <br> <br> <br> <br> <br> <br>
