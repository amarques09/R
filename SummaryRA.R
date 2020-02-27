############Data.frame

##indexation
#numeric

###########Loops


#### Create a loop that iterates through all the positions of 'v' and, for each
#### iteration, prints "FALSE value found" if the value of 'v' in that position
#### is FALSE. Do this using both a for and a while loop. Each loop should contain an if
#### (or if-else) clause.

v <- sample(c(TRUE, FALSE), 100, replace = TRUE);

for (index in 1:length(v)){
  if (!v[index]){
    print("FALSE value found")
  } else {
  }
}


index <- 1;
while (index <= length(v)){
  if (!v[index]){
    print("FALSE value found")
  } else {
  }
  index <- index + 1;
}


#### Your objective now is to create a loop (you can choose which type) that iterates through the positions
#### of 'v' until it finds an index where 'v[index]' is distinct from 'v[index+1]', in that moment you should
#### print the value of 'index' and stop the loop using the "break" command. Beware infinite loops.

v <- sample(c(TRUE, FALSE), 100, replace = TRUE);

index;
for (index in 1:length(v)){
  if (v[index] != v[index+4]){
    print(index);
    break;
  } 
}


#### (*) Exercise 5: Create a while clause that goes through the values of 'v' ####
#### printing them until it finds a 'v[index]' between 9 and 10 (both included).
#### At that moment, print "Excellent score found at position FILL", where FILL
#### is the position of 'v' where you find this value, and stop the loop. 
#### Suggestion: use break.

index <- 1;

while (index <= length(v)){
  print(v[index]);
  if (v[index] >= 9 & v[index] <= 10){
    print(sprintf("Excellent score found at position %s", index));
    break;
  }
  index <- index + 1;
}


#### Code two nested for loops in order to add a different random value
#### to each cell of the matrix. Suggestion: use runif to compute
#### each random number.

new_M <- matrix(0, nrow = nrow(M), ncol = ncol(M));

for (i in 1:nrow(new_M)){
  for (j in 1:ncol(new_M)){
    M[i,j] = M[i,j] + runif(1, 0, 1);
  }
}

identical(M, new_M);
head(M);
head(new_M);

#######Matrix

df <- matrix(c("Rex", "Tom", "Jerry", "Bobby",
               "dog", "cat", "mouse", "cat",
               3, 5, 2, 7), nrow = 4, ncol = 3);
colnames(df) <- c("name", "type", "age");
df;

##indexation
#numeric

### Use numeric indexation to get the value of column 'type' in the third row.

df[3, 2];

#logical

### Use logical indexation to get the rows where 'type' is equal to cat. Do not create
### the logical index manually, it should be the output of a logical operator (>, <, ==, !=, etc.)
df[df[, colnames(df) == "type"] == "cat",];

#byname

### Use indexation by name to get the age of Tom. You can select the row of 'df' 
### corresponding to Tom using any indexation method, but the age column must be 
### selected using indexation by name.

df[2, "age"];

###########FUNCTIONS

#### (*) EXERCISE 5: Create a function, called 'my_or', that returns as output  ####
#### TRUE only if the first argument, named 'x', or the second argument, named 'y', 
#### are TRUE. Use default values so that the output of

my_or();


#### is FALSE

my_or <- function(x = FALSE, y = FALSE) {
  result <- x | y;
  return(result);
}


###########Data.table

#Creating a data.table
dt1 <- data.table(v1=v1,v2=v2);

#Converting to data.table
dt <- as.data.table(iris);

# data.table using dt[i, j, by] operator
res_3 <- dt[Sepal.Length >= 5.4 & Petal.Length <= 2.0];
res_3;

#counting the number of rows with Petal.Length greater than 5 using 
dt[Petal.Length > 5, list(n= .N)];
#Counting the number of cars (rows) with 6 cylinders
dt[ cyl ==6, list(n = .N)]

#creating a new column and summing two rows 
dt[Sepal.Length >= 5.4, list(sum_width = sum(Sepal.Width + Petal.Width))];

#computing mean for each value of Species.
dt[, list(mean_sepallenght = mean(Sepal.Length)), by = "Species"]
dt[, list(avg_sepal_length = mean(Sepal.Length)), by = "Species"]

#compute the maximum weight for each value of gear.
dt[,list(max_w = max(wt)),by = "gear" ]

#computing the number of something, grouped by x, wiht a condition on the rows (>4);
dt[Petal.Length > 4,list(n=sum(Sepal.Length>5)), by = "Species"];

#using all the 3 [i,j,by]

#One variable
dt[Sepal.Length > 5, list(n = .N), by = "Species"];
res <- dt[Sepal.Length > 5, list(n = .N), by = c("Species", "Sepal.Length")];

#Multiple variables
res <- dt[Sepal.Length > 5, list(n = .N, avg_sepal_widgt = mean(Sepal.Width)),
          by = c("Species", "Sepal.Length")];
View(res);

dat <-readRDS ("/Users/amand/Documents/IE/Programming R/Quizz2/userbase.RData");
head(dat[, list(n= .N), by = "origin"][order(n, decreasing = TRUE)],1);
head(dat[, list(n = .N), by = "destination"][order(n, decreasing = TRUE)],5);

#compute by number of cylinders, rows of automatic transmission and manual. Only cars with more than 3 gears.
dt[ gear > 3, list(automatic= sum(am == 0), manual = sum(am ==1) ) , by = "cyl"]

#compute the minimum value of 'mpg' for each value of 'am', taking into account only cars with more than 4 cylinders.
dt[cyl >4, min(.SD[,"mpg"]), by = "am"];

# Take into account only flights bought after "2018-11-01"
head(dat)
dat[bookind_date>=as.Date("2018-11-01")];

###Chaining

dt[Sepal.Length > 5, list(avg_sepal_width = mean(Sepal.Width)), 
   by = "Species"][order(avg_sepal_width)]

# Get the top 5 users in terms of number of flights purchased via online channels
head(dat[sale_channel=="online",list(n_purchases= .N), by = "user"][order(n_purchases,
                                                                          decreasing = TRUE)],5);
# Get also the top 5 in terms of price per purchase.
head(dat[canal_venta == "online", list(avg_price = mean(price)), by = "user"][order(avg_price,
                                                                                    decreasing = TRUE)], 5)

###.SD (applies function to all the dt)

my_mode <- function(x) {
  x <- as.factor(x);
  ret <- names(sort(table(x), decreasing = FALSE))[1];
  return(ret);
}
#Decreasing = FALSE because it wants the less frequent value;

dt <- as.data.table(mtcars);
less_frequent <- dt[, sapply(.SD, my_mode)];
names(less_frequent) <- colnames(mtcars);
less_frequent


my_percentile <- function(x){
  quantile(x, 0.75);
}

dt <- as.data.table(mtcars);
third_quartiles <- dt[, sapply(.SD, my_percentile)];
names(third_quartiles) <- colnames(mtcars);
third_quartiles


null_counter <- function(x) {
  ret <- sum(x > 5);
  return(ret);
}

dt <- as.data.table(mtcars);

large_values <- dt[, sapply(.SD, null_counter)];
names(large_values) <- colnames(mtcars);

###########Lists

###########Plots
#general

plot(v, col = "red", fg = "blue", pch = 2, type = "l",
     main = "My first plot", xlab = "x", ylab = "y", 
     xlim = c(10,90), ylim = c(20,100),
     lty = 2, font = 2, las = 1, cex.axis = 1);

# Table of frequencies
tf <- table(v);
barplot(tf);

#histogram
hist(v, col = "purple", fg = "red",
     main = "TITLE", sub= "subtitle",xlab = "x values",
     ylim = c(-10,30),
     font = 2, las = 2, cex.axis = 0.6,
     breaks = 5
);

#barplot

barplot(frequencies, col = "green", font = 2);

#boxplot

boxplot(v);
v <- c(v, 1000);
boxplot(v, outline = FALSE);

boxplot(v, col = "yellow", 
        main = "my_blox", sub = "subtitle", xlab = "x", 
        ylab = "value",
        ylim = c(-100,500),
        las = 3, cex.axis = 1,
        outline = FALSE);


#density
plot(density(v), type = "l", col = "blue");

# Replace missing values
sapply(dat, function(x){sum(is.na(x))});
dat$price[is.na(dat$price)] <- mean(dat$price, na.rm = TRUE);
sapply(dat, function(x){sum(is.na(x))});

# Plot density
plot(density(dat$price), col = "blue");

#cumulative distribution
plot(ecdf(v), col = "blue");


###########Ggplot

### Create plot
library(ggplot2)
my_plot <- ggplot(as.data.table(mtcars),
                  aes(x=1:nrow(mtcars),y=cyl));
my_plot <- my_plot + geom_point(col="purple"); (#layer)
my_plot <- my_plot + labs(subtitle = "From mtcars dataset", x = "index");
my_plot 


### Add layer to set palette
my_plot <- my_plot + 
  scale_colour_brewer(palette = "YlOrRd")
print(my_plot);


library(data.table);
library(ggplot2);

### Create plot
my_plot <- ggplot(as.data.table(mtcars), 
                  aes(x=1:nrow(mtcars), y = mpg));

### Add layer of points
my_plot <- my_plot + geom_line(col = "blue");


### Add layer to set axis ranges
my_plot <- my_plot + xlim(c(0, 50))


### Add layer to set main and axis titles
my_plot <- my_plot + 
  labs(title="Car mpg", subtitle="From mtcars dataset", 
       y="mpg", caption="mtcars dataset mpg");


### Print plot
my_plot;


### Create plot
my_plot <- ggplot(as.data.table(mtcars), 
                  aes(y = wt));

### Add layer of points
my_plot <- my_plot + geom_boxplot(col = "blue",
                                  fill = "green",
                                  outlier.colour ="red", notch=FALSE);


### Add layer to set main and axis titles
my_plot <- my_plot + 
  labs(subtitle="From mtcars dataset", 
       y="wt");

### Add layer to set axis ranges
my_plot <- my_plot +  ylim(c(0, 10))



### Print plot
my_plot;


### Create plot
my_plot <- ggplot(as.data.table(mtcars), 
                  aes(y = mpg));

### Add layer of points
my_plot <- my_plot + geom_boxplot(col = "purple",
                                  fill = "blue",
                                  outlier.colour ="red");


### Add layer to set main and axis titles
my_plot <- my_plot + 
  labs(subtitle="From mtcars dataset (again)", 
       y="mpg");

### Add layer to reverse y axis
my_plot <- my_plot + scale_y_reverse();


### Print plot
my_plot;

#############Mark-Down



##########Shiny

############################# UI ##################################
# Define UI for application that draws a histogram
shinyUI(fluidPage(  
  title = 'mtcars Analysis',
  
  
  mainPanel(
    img(src='logo.png', align = "right"),
    tags$head(
      tags$style(type='text/css',
                 ".nav-tabs {font-size: 25px} ")),
    tabsetPanel(
      type = "tabs",
      
      tabPanel("dt[i, j, by]", 
               
               ### SELECTORS
               
               # Slider
               sliderInput("selected_rows", label = h3("Rows Selected"), min = 1, 
                           max = nrow(mtcars), value = c(1, 10),
                           round = -2, step = 1),
               
               # Multiple response choice
               selectInput("selected_columns", label = h3("Columns Selected"),
                           choices = c("",colnames(mtcars)),
                           selected = 1,
                           width='55%',
                           multiple = TRUE),
               
               # Single response choice
               selectInput("selected_op", label = h3("Operation to Compute"),
                           choices = c("", "mean", "min", "max"),
                           selected = 1,
                           width='55%',
                           multiple = FALSE),
               
               
               # Multiple response choice
               selectInput("selected_group", label = h3("Group By"),
                           choices = c("",colnames(mtcars)),
                           selected = 1,
                           width='55%',
                           multiple = TRUE),
               
               ### TABLE 1
               
               # Print table
               fluidRow(
                 column(12, DT::dataTableOutput('data'))    
               ),
               
               # Download button
               sidebarPanel(
                 radioButtons("filetype", "File type:",
                              choices = c("excel", "csv", "tsv")),
                 downloadButton('downloadData', 'Download')
               )
               ,
               
               ### TABLE 2
               # Print table
               fluidRow(
                 column(12, DT::dataTableOutput('operation'))
               ),
               
               # Download button
               sidebarPanel(
                 radioButtons("filetype2", "File type:",
                              choices = c("excel", "csv", "tsv")),
                 downloadButton('downloadData2', 'Download')
               )
      ),
      
      tabPanel("Plot", 
               
               # Single response choice
               selectInput("selected_column_plot", label = h3("Column to Plot"),
                           choices = c("",colnames(mtcars)),
                           selected = 1,
                           width='55%',
                           multiple = FALSE),
               
               fluidRow(
                 column(12, plotOutput("plot"))
               )
               
      )
      
      

# Define UI for application that draws a histogram
      shinyUI(fluidPage(
        
        # Application title
        titlePanel("Show Data Set"),
        
        # Sidebar with a slider input for number of bins
        sidebarLayout(
          sidebarPanel(
            selectInput("dataset", label = h3("Select the data set"),
                        choices = c("iris","mtcars"),
                        selected = 1,
                        width='55%',
                        multiple = FALSE)
          ),
          
          # Show a plot of the generated distribution
          fluidRow(
            column(12, DT::dataTableOutput('tablechoosen'))
          )
        )
      ))
      
      
      
########################## Server

library(shiny);
library(data.table);
library(ggplot2);

# Define server logic required to draw a histogram
shinyServer(function(input, output){
  
  ### Reactive functions
  
  # Function to filter data
  compute_data <- reactive({
    if (length(input$selected_columns) > 0){
      dat <- as.data.table(mtcars[input$selected_rows[1]:input$selected_rows[2], input$selected_columns]);
    }else{
      dat <- data.table();
    }
    
    return(dat)
  })
  
  # Function to compute operation over filtered data
  compute_operation <- reactive({
    dat <- compute_data();
    if (input$selected_op!=""){
      if (input$selected_op == "mean"){
        dat <- dat[, sapply(.SD, mean), by = eval(input$selected_group), 
                   .SDcols = setdiff(input$selected_columns,input$selected_group)];
      } else if (input$selected_op == "max"){
        dat <- dat[, sapply(.SD, max), by = eval(input$selected_group), 
                   .SDcols = setdiff(input$selected_columns,input$selected_group)];
      } else if (input$selected_op == "min"){
        dat <- dat[, sapply(.SD, min), by = eval(input$selected_group), 
                   .SDcols = setdiff(input$selected_columns,input$selected_group)];
      }
    }else{
      dat <- data.table();
    }
    
    return(dat)
  })
  
  ### Tables
  
  # Print 'data' table
  output$data = DT::renderDataTable(
    compute_data(), filter = 'top', rownames=FALSE)
  
  # Print 'operation' table
  output$operation = DT::renderDataTable(
    compute_operation(),  filter = 'top',  rownames=FALSE)
  
  ### Download button
  
  # 'data' table button
  output$downloadData <- downloadHandler(
    
    # This function returns a string which tells the client
    # browser what name to use when saving the file.
    filename = function() {
      paste("filtered_data", gsub("excel", "csv",input$filetype), sep = ".")
    },
    
    # This function should write data to a file given to it by
    # the argument 'file'.
    content = function(file) {
      if (input$filetype == "excel"){
        write.csv2(compute_data(), file);
      } else {
        sep <- switch(input$filetype, "csv" = ",", "tsv" = "\t");
        write.table( compute_data(), file, sep = sep,
                     row.names = FALSE);
      }
    }
  )
  
  # 'operation' table button
  output$downloadData2 <- downloadHandler(
    
    # This function returns a string which tells the client
    # browser what name to use when saving the file.
    filename = function() {
      paste("operation", gsub("excel", "csv",input$filetype2), sep = ".")
    },
    
    # This function should write data to a file given to it by
    # the argument 'file'.
    content = function(file) {
      if (input$filetype2 == "excel"){
        write.csv2(compute_operation(), file);
      } else {
        sep <- switch(input$filetype2, "csv" = ",", "tsv" = "\t");
        write.table( compute_operation(), file, sep = sep,
                     row.names = FALSE);
      }
    })
  
  ### Plots
  compute_plot <- reactive({
    dat <- compute_data();
    if (input$selected_column_plot != ""){
      ### Create plot
      my_plot <- ggplot(dat, aes(x=1:nrow(dat),
                                 y = as.numeric(t(dat[,input$selected_column_plot, with = F]))));
      
      ### Add layer of points
      my_plot <- my_plot + geom_point(col = "red");
      
      
      ### Add layer to set main and axis titles
      my_plot <- my_plot +
        labs(title="Scatterplot", subtitle="From mtcars dataset",
             y= input$selected_column_plot, x="n", caption="mtcars dataset");
      
      
      ### Add layer to set axis values
      my_plot <- my_plot +  scale_x_continuous(breaks=seq(0, nrow(dat), 2));
      
      ### Add layer to change theme
      my_plot <- my_plot + theme_dark();
      my_plot;
    }
  })
  
  output$plot <- renderPlot({
    compute_plot();
  })
  
}
)     


library(shiny)
library(data.table)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
  compute_data <- reactive({
    if (input$dataset == "iris"){
      dat <- data.table(iris);
    } else {
      dat <- data.table(mtcars);
    }
    return(dat)
  });
  
  # Print table
  output$tablechoosen = DT::renderDataTable(compute_data(), filter = 'top', rownames=FALSE)
  
})



#########################Reading R Objects

#create a file path
#readRDS
#readCSV
#read.xlsx
#fread

##########################DATA CLEANING
#Number of unique values in each column of dat
sapply(dat, function(x){length(unique(x))});

#Number of constant variables
constant_variables <- function(dat){
  n_unique_values <- sapply(dat, function(x){length(unique(x))});
  constant_variables <- names(n_unique_values)[n_unique_values == 1];
  return(constant_variables);
}

## Let?s call this function to detect constant variables in dat
constant_var <- constant_variables(dat);
constant_var;

# Remove these variables from the dataset
dat <- dat[, setdiff(colnames(dat), constant_var)];
colnames(dat);

#####missing values

#check the number of missing values in each column of dat
sapply(dat, function(x){sum(is.na(x))}); 

# In percentages
sapply(dat, function(x){100*sum(is.na(x))/length(x)});

###replacing it with mean and median

mu <- mean(dat$price, na.rm = TRUE);
dat$price[is.na(dat$price)] <- mu;
sapply(dat, function(x){sum(is.na(x))});

#mode
my_mode <- function(x) {
  x <- as.factor(x);
  ret <- names(sort(table(x), decreasing = TRUE))[1];
  return(ret);
}


# replace with mean or mode depending on class
dat$price[sample(1:nrow(dat), 50)] <- NA;
dat$origin_airport[sample(1:nrow(dat), 50)] <- NA;
sapply(dat, function(x){sum(is.na(x))}); 

fill_missing_values <- function(x){
  if (class(x) == "numeric" | class(x) == "integer"){
    x[is.na(x)] <- mean(x, na.rm = TRUE);
  } else {
    x <- as.factor(x);
    x[is.na(x)] <- my_mode(x);
  }
  return(x);
}

dat_2 <- data.frame(sapply(dat, fill_missing_values));
dim(dat);
dim(dat_2);
sapply(dat, function(x){sum(is.na(x))}); 
sapply(dat_2, function(x){sum(is.na(x))}); 


#### Create a function to compute the frequency (in percentage) ####
#### of the mode or most frequent value in a vector. Call this function for each
#### column in 'df'

my_mode <- function(x){
  x <- as.factor(x)
  ret <- names(sort(table(x), decreasing = TRUE))[1];
  return(ret); }

main_frequent <- function(x){
  x <- as.character(x);
  return(100*sum(x== my_mode(x), na.rm = TRUE)/length(x));
}


sapply(df,main_frequent)

##########LINEAR REGRESSION

# Get model predictions
predictions <- predict(model_l1_all, newdata = dat);
manual_predictions <- dat$wt*model_l1_all$coefficients[2] + model_l1_all$coefficients[1] 
head(predictions);
head(manual_predictions)

##Train/test
# setting seed to reproduce results of random sampling
set.seed(100); 

# row indices for training data (70%)
train_index <- sample(1:nrow(dat), 0.7*nrow(dat));  

# training data
train <- dat[train_index]; 

# test data
test  <- dat[-train_index ]

dim(dat);
dim(train);
dim(test);

#### 1 variable model using train/test

# build linear regression model to predict mpg using as predictor variable wt
model_l1_train <- lm(mpg ~ wt, data = train);

# Get model predictions for train
predictions_train <- predict(model_l1_train, newdata = train);

# Get model predictions for test
predictions_test <- predict(model_l1_train, newdata = test);

# Get errors
errors_train <- predictions_train - train$mpg;
errors_test <- predictions_test - test$mpg;

# Compute Metrics
mse_train <- round(mean(errors_train^2), 2);
mae_train <- round(mean(abs(errors_train)), 2);

mse_test <- round(mean(errors_test^2), 2);
mae_test <- round(mean(abs(errors_test)), 2);

# Build comparison table
comp <- data.table(model = c("lm_1var"), 
                   mse_train = mse_train, mae_train = mae_train,
                   mse_test = mse_test, mae_test = mae_test);
comp;



#### All variables
    
    # build linear regression model to predict mpg using all the other variables as predictors
    model_lall <- lm(mpg ~ ., data = train);
    
    
    # Check model info
    print(model_lall); # mpg = w1*cyl + w2*disp + ... + w*carb + (Intercept)
    summary(model_lall);
    summary(model_l1_train);
    
    # Get model predictions
    predictions_train <- predict(model_lall, newdata = train);
    predictions_test <- predict(model_lall, newdata = test);
    
    # Get errors
    errors_train <- predictions_train - train$mpg;
    errors_test <- predictions_test - test$mpg;
    
    # Compute Metrics
    mse_train <- round(mean(errors_train^2), 2);
    mae_train <- round(mean(abs(errors_train)), 2);
    
    mse_test <- round(mean(errors_test^2), 2);
    mae_test <- round(mean(abs(errors_test)), 2);
    
    # Build comparison table
    comp <- rbind(comp,
                  data.table(model = c("lm_allvar"), 
                             mse_train = mse_train, mae_train = mae_train,
                             mse_test = mse_test, mae_test = mae_test));
    comp;
    
    
hyperparameter optimization ############################

### Check svm hyperparameters
help(svm);

### Define grid
c_values <- 10^seq(from = -3, to = 3, by = 1);
eps_values <- 10^seq(from = -3, to = 0, by = 1);
gamma_values <- 10^seq(from = -3, to = 3, by = 1);

### Compute grid search
grid_results <- data.table();

for (c in c_values){
  for (eps in eps_values){
    for (gamma in gamma_values){
      
      print(sprintf("Start of c = %s - eps = %s - gamma = %s", c, eps, gamma));
      
      # train SVM model with a particular set of hyperparamets
      model <- svm(mpg ~ ., data = train, kernel="radial",
                   cost = c, epsilon = eps, gamma = gamma);
      
      # Get model predictions
      predictions_train <- predict(model, newdata = train);
      predictions_val <- predict(model, newdata = val);
      
      # Get errors
      errors_train <- predictions_train - train$mpg;
      errors_val <- predictions_val - val$mpg;
      
      # Compute Metrics
      mse_train <- round(mean(errors_train^2), 2);
      mae_train <- round(mean(abs(errors_train)), 2);
      
      mse_val <- round(mean(errors_val^2), 2);
      mae_val <- round(mean(abs(errors_val)), 2);
      
      # Build comparison table
      grid_results <- rbind(grid_results,
                            data.table(c = c, eps = eps, gamma = gamma, 
                                       mse_train = mse_train, mae_train = mae_train,
                                       mse_val = mse_val, mae_val = mae_val));
    }
  }
}

# Order results by increasing mse and mae
grid_results <- grid_results[order(mse_val, mae_val)];

# Check results
View(grid_results);
grid_results[1];
grid_results[which.max(mse_train)]; # Underfitting! High bias-low variance (Bias-Variance tradeoff)

# Get optimized hyperparameters
best <- grid_results[1];
best;


### Train final model
# train SVM model with best found set of hyperparamets
model <- svm(mpg ~ ., data = rbind(train,val), kernel="radial",
             cost = best$c, epsilon = best$eps, gamma = best$gamma);

# Get model predictions
predictions_train <- predict(model, newdata = train);
predictions_val <- predict(model, newdata = val);
predictions_test <- predict(model, newdata = test);

# Get errors
errors_train <- predictions_train - train$mpg;
errors_val <- predictions_val - val$mpg;
errors_test <- predictions_test - test$mpg;

# Compute Metrics
mse_train <- round(mean(errors_train^2), 2);
mae_train <- round(mean(abs(errors_train)), 2);

mse_val <- round(mean(errors_val^2), 2);
mae_val <- round(mean(abs(errors_val)), 2);

mse_test <- round(mean(errors_test^2), 2);
mae_test <- round(mean(abs(errors_test)), 2);

# Build comparison table
comp <- rbind(comp,
              data.table(model = c("optimized_svm"), 
                         mse_train = mse_train, mae_train = mae_train,
                         mse_test = mse_test, mae_test = mae_test));
comp; # Best model in terms of error metrics!! More accuracy (Accuracy-'explainability' tradeoff)


##########Classification

#Split Train and Test


# setting seed to reproduce results of random sampling
set.seed(57); 

# row indices for training data (70%)
train_index <- sample(1:nrow(dat), 0.7*nrow(dat));  

# training data
train <- dat[train_index]; 

# test data
test  <- dat[-train_index ]

#Regression

table(train$am);
class(train$am);

# build linear classifier model to predict am using all the other variables as predictors
model <- glm(am ~ ., data = train);


# Check model info
print(model); 

# Get model predictions
predictions_train_num <- predict(model, newdata = train);
predictions_test_num <- predict(model, newdata = test);
head(predictions_train_num); 
predictions_train <- as.numeric(predictions_train_num >= 0.5);
predictions_test <- as.numeric(predictions_test_num >= 0.5);

# Accuracy
acc_train <- 100*sum(predictions_train == train$am)/nrow(train);
acc_test <- 100*sum(predictions_test == test$am)/nrow(test);

# Build comparison table
comp <- data.table();
comp <- rbind(comp,
              data.table(model = c("regression_glm"), 
                         acc_train = acc_train, acc_test = acc_test));
comp;

##hyperparameter optimization

### Define grid
c_values <- 10^seq(from = -3, to = 3, by = 1);
gamma_values <- 10^seq(from = -3, to = 3, by = 1);

# epsilon parameter only for regression
# eps_values <- 10^seq(from = -3, to = 3, by = 1);

### Compute grid search
grid_results <- data.table();

for (c in c_values){
  for (gamma in gamma_values){
    
    print(sprintf("Start of c = %s - eps = %s", c, eps));
    
    # train SVM model with a particular set of hyperparamets
    model <- svm(am ~ ., data = train, kernel="radial",
                 cost = c, gamma = gamma);
    
    # Get model predictions
    predictions_train <- predict(model, newdata = train);
    predictions_val <- predict(model, newdata = val);
    
    # Accuracy
    acc_train <- 100*sum(predictions_train == train$am)/nrow(train);
    acc_test <- 100*sum(predictions_test == test$am)/nrow(test);
    
    
    # Build comparison table
    grid_results <- rbind(grid_results,
                          data.table(c = c, eps = eps, gamma = gamma, 
                                     acc_train = acc_train, acc_test = acc_test));
  }
}


# Order results by decreasing accuracy
grid_results <- grid_results[order(-acc_test, -acc_train)];

# Check results
View(grid_results);

# Get optimized hyperparameters
best <- grid_results[1];
best;


### Train final model
# train SVM model with best found set of hyperparamets
model <- svm(am ~ ., data = train, kernel="radial",
             cost = best$c, epsilon = best$eps);

# Get model predictions
predictions_train <- predict(model, newdata = train);
predictions_val <- predict(model, newdata = val);
predictions_test <- predict(model, newdata = test);

# Accuracy
acc_train <- 100*sum(predictions_train == train$am)/nrow(train);
acc_val <- 100*sum(predictions_val == val$am)/nrow(val);
acc_test <- 100*sum(predictions_test == test$am)/nrow(test);

# Build comparison table
comp <- rbind(comp,
              data.table(model = c("optimized_svm"), 
                         acc_train = acc_train, acc_test = acc_test));
comp;

    
######## SEGMENTATION / CLUSTERING
    
library(data.table);
root_path <- "/Users/amand/Documents/IE/Programming R/18/";
set.seed(14);

# We have missing values
sapply(dat, function(x){sum(is.na(x))});

# Remove missing values (using mean)
dat$price[is.na(dat$price)] <- mean(dat$price, na.rm = TRUE);

# Perform k-means (again)
clustering <- kmeans(x = dat[, c("price", "ant")], centers = 2);

# Check result
class(clustering);
str(clustering);
table(clustering$cluster);

##### k prototype

# kproto only accepts numeric and factors
non_numeric <- colnames(dat)[sapply(dat, class) != "numeric"];
dat[, non_numeric] <- data.table(sapply(dat[, non_numeric, with = F], as.factor),
                                 stringsAsFactors = TRUE);
#casting them to factors
sapply(dat, class);

# Perform k-prototypes
clustering <- kproto(x = dat, k = 4);

# Visualize clusters
clusplot(dat, clustering$cluster, color=TRUE, shade=TRUE, 
         labels=0, lines=0, main = "k-prototypes");
clprofiles(object = clustering, x = dat, vars = colnames(dat));


#PCA

# Cast all to numeric
dat[, non_numeric] <- data.table(sapply(dat[, non_numeric, with = F], as.numeric));

# Perform PCA
prin_comp <- prcomp(dat, center = TRUE, scale. = TRUE);

# Check results
class(prin_comp);
str(prin_comp);

# Get dataset after PCA
pca_dat <- prin_comp$x # method 1
pca_dat_2 <- data.table(predict(prin_comp, newdata = dat)); # method 2
head(pca_dat);
head(pca_dat_2);

# Choose number of variables manually
number_of_variables <- 2;
final_pca_dat <- pca_dat[,1:number_of_variables];
head(final_pca_dat);
plot(final_pca_dat);

# Choose number of variables using explained variance
summary(prin_comp);
prop_variance_explained <- summary(prin_comp)$importance[3,];
threshold_variance <- 0.8;
number_of_variables <- min(which(prop_variance_explained > threshold_variance))
final_pca_dat <- pca_dat[,1:number_of_variables];
head(final_pca_dat);

###### Recommendation System

# Interaction matrix
flights <- matrix(c(0, 0, 1, 1, 4, 0, 0,
                    1, 0, 0, 0, 0, 1, 2,
                    1, 2, 0, 0, 0, 0, 2,
                    1, 2, 0, 2, 1, 0, 1,
                    0, 3, 2, 1, 3, 0, 0), nrow = 7, ncol = 5);
colnames(flights) <- c("santiago", "sevilla", "tenerife", "valencia", "vigo");
rownames(flights) <- paste0("user_",1:7);
flights;

# Similarity function (cosine)
cosine <- function(x, y){
  as.numeric(x%*%y) / (norm(x, "2")*norm(y, "2"));
}

x <- c(1, 2, 3);
y <- c(3, 0, 1);

cosine(x, x);
cosine(x, x+1);
cosine(x, y);
cosine(x, -x);
plot(cosine(x+1,x+y))
# Similarity between two items (routes)
flights[, "santiago"];
flights[, "vigo"];
cosine(flights[, "santiago"], flights[, "vigo"]);

# Similarity matrix
sim <- matrix(rep(0, length(flights)), nrow = ncol(flights), ncol = ncol(flights));
rownames(sim) <- colnames(flights);
colnames(sim) <- colnames(flights);
sim;

for (i in 1:ncol(flights)){
  for (j in 1:ncol(flights)){
    sim[i, j] <- cosine(flights[, i], flights[, j]);
  }
}
sim;

# Interests of an user
flights["user_1",];
sim["santiago",]
flights["user_1",]%*%sim["santiago",] # Interest in Santiago

flights["user_1",];
sim["valencia",]
flights["user_1",]%*%sim["valencia",] # Interest in Valencia

interests <- c();
for (i in 1:ncol(flights)){
  interests <- c(interests, flights["user_1",]%*%sim[,i])
}
names(interests) <- colnames(flights);
interests;
flights["user_1",];
names(interests)[which.max(interests)]; # We should recommend him Tenerife

# Interests of all users
interests <- matrix(rep(0, length(flights)), nrow = nrow(flights), ncol = ncol(flights));
rownames(interests) <- rownames(flights);
colnames(interests) <- colnames(flights);
interests;

for (i in 1:nrow(interests)){
  for (j in 1:ncol(interests)){
    interests[i, j] <- flights[i,]%*%sim[,j]
  }
}
interests;

# "Best" recommendation for each user
recommendations <- apply(interests, 1, which.max);
recommendations;
recommendations <- colnames(flights)[recommendations];
recommendations
names(recommendations) <- rownames(flights);
recommendations;

flights;





