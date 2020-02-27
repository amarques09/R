### This is the final exam of PROGRAMMING R.

### This exam consists of ten questions with three levels of difficulty, ranging from
### basic (*) to advance (***). Each question is worth 1 point.

### In the exercises where a final answer is requested (these questions will be indicated by
### the expression "(QUESTION X)", where X is the corresponding number of the question), 
### please fill the line "### ANSWER TO X:", which you will find at the end of the exercise 
### where QUESTION X was formulated, with the corresponding answer (there can be more than
### one QUESTION in the same exercise).

### Good luck!
######################################################################################################

#### (*) Exercise 1: Together with this quiz you received two files (in folder "files"). Read both ####
#### of them using the corresponding functions and store them into two different variables. Bear in mind that
#### both files represent the same dataset so the corresponding variables should be identical. Do not use fread.


# This line should output TRUE
identical(file1, file2);

library(data.table);
folder_path <- "C:/Users/amand/Documents/IE/Programming R/RealFinalQuizz/files";

file1 <- readRDS(file.path(folder_path, "file1.rds"));
file1;
file2 <- read.table(file.path(folder_path, "file2.csv"), sep = "_", header = TRUE);
file2
file1
file2;

identical(file1,file2);

#### (*) Exercise 2: You have the following data.frame ####

df <- data.frame(concept_name = c("data.table", "kmeans", "EDA", "SVM"),
                 complexity = c(9, 2, 5, 10),
                 usefulness = c(10, 4, 7, 10));
df;
#### Use any indexation method to get the complexity of kmeans
df[2,]$complexity;

### Use indexation by variable ($ operator) in combination with logical indexation
### to get the name of concepts with usefulness equal to 10.

df[df$usefulness == 10,]$concept_name;


#### (*) Exercise 3: Cast the mtcars dataset to a data.table. Then, compute the total sum of weight and displacement ####
#### for each number of carburetors, taking into account only rows corresponding to cars with less than 8 cylinders.
#### You should get one final number for each value of carburetors, corresponding to the total sum of weight and displacement
#### for all the cars with that particular value of carburetors. Use only one command with the format dt[i, j, by]
#### (maybe you do not need to use all of the three dimensions or indexes).

dt <- as.data.table(mtcars);
head(mtcars);

dt[carb < 4, c("wt","disp"),
   list(sum("wt","disp")
   by = "carb"];


#### (*) Exercise 4: You have the following vector ####

set.seed(140);
v <- sample(1:100, 100, replace = TRUE);

#### Create a while loop that iterates through all the values of 'v' printing the corresponding value when v[index] is
#### greater or equal than 50 and stopping the loop using BREAK when the value is less than 50. Use an if-else clause.


index <- 1;

#while (index <= length(v)){
  print(v[index]);
  if (v[index] >= 50){
    print(v[index]);
  } else {
        index < 50;
      break;
    } 
  }

#### (*) Exercise 5: Try to replicate the plot given in final_quiz_plot.png together  #### 
#### with this quiz using ggplot functions. Bear in mind that the values plotted come 
#### from the mtcars dataset.

library(ggplot2)

my_plot <- ggplot(as.data.table(mtcars), 
                  aes(x=1:nrow(mtcars),y=cyl));

### Add layer of points
my_plot <- my_plot + geom_histogram(col = "blue",
                                    fill = "blue" );

my_plot <- my_plot + 
  labs(subtitle="Mtcars dataset mpg", 
       x="mpg",
       y = "Frequency");
my_plot <- my_plot + scale_x_reverse();
my_plot;

#### (**) Exercise 6: Create a new R Markdown called "final_quiz_Rmarkdown" and ####
#### make the following modifications:  

#### 1) Modify the header to include a new parameter called 'column' with a default
#### value of "mpg".
#### 2) Modify the general configuration chunk so for any R block
#### the results and code are not printed by default.
#### 3) Create a new text block at the bottom that prints "mtcars has x rows" where x
#### is the actual number of rows of mtcars computed using inline code(i.e, computing r code
#### inside a text block). 
#### 4) Create a new code block below the previous text block to print from mtcars the
#### mean of the column indicated by the value of 'column'. The result and the code of this block must be shown
#### 5) Use the render function to run your Rmarkdown changing the column parameter
#### to "wt"

#### You have to submit the modified .Rmd together with this .R file.


#### (**) Exercise 7: Compute k-prototypes on iris to split the data into 3 clusters. ####
#### Add a new column to the iris dataset called "cl" indicating for each observation the cluster
#### to which it was assigned by k-prototypes. Compare the table of frequencies of "cl" and "Species" 
#### to search for similarities between them.

library(clustMixType);
dat <- iris
# kproto only accepts numeric and factors
# Check classes
sapply(dat, class);
clustering <- kproto(x = dat, k = 3);
clusplot(dat, clustering$cluster, color=TRUE, shade=TRUE, 
         labels=0, lines=0, main = "k-prototypes");

head(cbind(dat, data.table(cl = clustering$cluster))); 


#### (**) Exercise 8: Create a new shiny app called "final_quiz_shiny" and make the  ####
#### following modifications to the ui.R and server.R (or app.R) files 

#### 1) Create a new selectInput called 'dataset' where an user can
#### choose between mtcars or mpg dataset (from gpplot2 library) and a new slider called 'rows'
#### to select a range of rows of that dataset.
#### 2) Use the value of this inputs to plot the rows selected in 'rows' of the first column
#### from the data selected in 'dataset'. You can choose the type of plot but bear in mind that mpg
#### dataset has character columns.
#### 3) If a user changes the value of 'dataset' or 'rows' selected the plot should
#### also automatically change.
#### 4) You can show this plot at any position in your shiny app.


#### You have to submit this shiny app together with this .R file.



#### (***) Exercise 9: You have the following data.frame

set.seed(140);
df <- data.frame(V1 = sample(letters, 50, replace = TRUE),
                 V2 = sample(letters, 50, replace = TRUE),
                 V3 = sample(letters, 50, replace = TRUE),
                 stringsAsFactors = FALSE);
head(df);

#### Now check this block of code

new_df <- df;
for (col in colnames(df)){
  values <- df[, col];
  frequencies <- 100*table(values)/length(values);
  values_to_group <- names(frequencies[frequencies <= 5]);
  values[values %in% values_to_group] <- "OTHERS";
  new_df[, col] <- values;
}
head(new_df);

#### Carry out these steps:

#### 1) Create a function 'group_categories' to compute the same operations for
#### a given vector as each iteration of the previous loop is doing for each 
#### column of df. This should return the input vector with categories grouped

group_categories <- function(x){
  correlation <- abs(cor(as.numeric(x), mtcars$wt));
  if (correlation > 0.75){
    ret <- TRUE;
  } else {
    ret <- FALSE;
  }
  return(ret);
}



#### 2) Use this function together with any vectorization method to get 
#### the same output as the one in 'new_df'

new_df_2 <- data.frame(sapply(df, fill_missing));
new_df;
new_df_2;

#### 3) Try to generalize your function so this operation can be called
#### with any choice of frequency threshold to group other than 5, which
#### should remain the default value for the threshold.

sapply(data.frame(sapply(df, group_categories)), function(x){length(unique(x))});

#### should give the following output

# V1 V2 V3 
# 8  8  9

#### and

sapply(data.frame(sapply(df, group_categories, 10)), function(x){length(unique(x))});

#### should give the following output

# V1 V2 V3 
# 1  1  2 



#### (***) Exercise 10: Your goal now is to predict the 'color' column of diamonds (from ggplot2 library) ####
#### dataset. We will only use a subsampling of 2000 rows where 'color' is E or G, i.e

library(data.table);
library(ggplot2);

# setting seed to reproduce results of random sampling. 
set.seed(14); 

# Filter data
dat <- data.table(diamonds);
dat <- dat[color %in% c("E", "G")];
dat <- droplevels(dat);
index_selected <- sample(1:nrow(dat), 2000);
dat <- dat[index_selected];

# row indices for training data (70%)
train_index <- sample(1:nrow(dat), 0.7*nrow(dat));  

# row indices for validation data (15%)
val_index <- sample(setdiff(1:nrow(dat), train_index), 0.15*nrow(dat));  

# row indices for test data (15%)
test_index <- setdiff(1:nrow(dat), c(train_index, val_index));

# split data
train <- dat[train_index]; 
val <- dat[val_index]; 
test  <- dat[test_index];

dim(dat);
dim(train);
dim(val);
dim(test);


#### Using train, val, and test find a model with more than 0.8 of AUC for the test set.
#### Suggestion: Use pROC library to compute AUC and save your script before training any model.

# Start cluster
library(foreach);
library(doParallel);
stopImplicitCluster();
registerDoParallel(cores = detectCores());

install.packages("pROC")
library(pROC);
AUC <- function(target, prediction){
  roc_curve <- roc(target, prediction);
  return(auc(roc_curve));
}

### Define grid
c_values <- c(seq(from = 10^1, to = 10^3, length.out = 10));
gamma_values <- c( seq(from = 10^-2, to = 1, length.out = 10));

### Compute grid search
grid_results <-  foreach (c = c_values, .combine = rbind)%:%
  foreach (gamma = gamma_values, .combine = rbind)%dopar%{
    library(e1071);
    library(data.table);
    library(pROC);
    
    print(sprintf("Start of c = %s - gamma = %s", c, gamma));
    
    # train SVM model with a particular set of hyperparamets
    model <- svm(cut ~ ., data = train, kernel="radial",
                 cost = c,  gamma = gamma, probability = TRUE);
    
    # Get model predictions
    predictions_train <- attr(predict(model, newdata = train, probability = TRUE),
                              "probabilities")[,1];
    predictions_val <- attr(predict(model, newdata = val, probability = TRUE),
                            "probabilities")[,1];
    
    # Compute Metrics
    auc_train <- AUC(train$cut, predictions_train);
    auc_val <- AUC(val$cut, predictions_val);
    
    # Build comparison table
    data.table(c = c, gamma = gamma, 
               auc_train = auc_train,
               auc_val = auc_val);
  }

# Order results by increasing mse and mae
grid_results <- grid_results[order(-auc_val, -auc_train)];

# Check results
best <- grid_results[1];

### Train final model
# train SVM model with best found set of hyperparamets
model <- svm(cut ~ ., data = train, kernel="radial",
             cost = best$c,  gamma = best$gamma, probability = TRUE);

# Get model predictions
predictions_train <- attr(predict(model, newdata = train, probability = TRUE),
                          "probabilities")[,1];
predictions_val <- attr(predict(model, newdata = val, probability = TRUE),
                        "probabilities")[,1];
predictions_test <- attr(predict(model, newdata = test, probability = TRUE),
                         "probabilities")[,1];

# Compute Metrics
auc_train <- AUC(train$cut, predictions_train);
auc_val <- AUC(val$cut, predictions_val);
auc_test <- AUC(test$cut, predictions_test);

## Summary
sprintf("AUC_train = %s - AUC_val = %s - AUC_test = %s", auc_train, auc_val, auc_test);


View(diamonds)
