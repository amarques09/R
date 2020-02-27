#########################################################################################################################

library(data.table);
### (*) EXERCISE 1: Create a variable of each one of the classes or data types we have
### seen during the course (feel free to choose its value). 
#### Do not use casting, create a new variable for each data type instead.

# 1.1 Numeric
v <- 3;

# 1.2 Integer
v <- 3L;

# 1.3 Character
v <- "3";

# 1.4 Logical
v <- TRUE;

# 1.5 Factor
v <- factor("dog");

# 1.6 Date (use as.Date or strptime)
v <- as.Date("2018-11-08");

# 1.7 Datetime/POSIXct (use as.POSIXct or strptime)
v <- as.POSIXct("2018-11-08 15:30:00");

# 1.8 Vector
v <- c(1, 2, 3);

# 1.9 Matrix
v <- matrix(1:4, nrow = 2, ncol = 2, byrow = TRUE);

# 1.10 Data.frame
v <- data.frame(v1 = c(1, 2), v2 = c("1", "2"));

# 1.11 Data.table
v <- data.table(v1 = c(1, 2), v2 = c("1", "2"));

# 1.12 List
v <- list(v1 = c(1, 2), v2 = c("1", "2"));


### (*) EXERCISE 2: You have the following vector:

v1 <- c(0, 1, 2);

### This vector belongs to one of the five basic variable classes we have seen during the
### course, which one? (QUESTION 1). 
### Cast 'v1' to each one of the other 4 variable classes in such a way that you create a new variable
### with each casting: v2, v3, v4 and v5.
### Then create a matrix with 'v1' as its first column, 'v2' as its second column, etc.
### To which R class belong each column of this matrix? (QUESTION 2).

# Castings
v2 <- as.integer(v1);
v3 <- as.character(v1);
v4 <- as.logical(v1);
v5 <- as.factor(v1);

# Create matrix
M <- matrix(c(v1, v2, v3, v4, v5), nrow = 3, ncol = 5); 
class(M[,1]);

### ANSWER TO 1: numeric
### ANSWER TO 2: character 



### (*) EXERCISE 3: You have the following data.frame:

df <- data.frame(name = c("Rex", "Tom", "Jerry", "Bobby"),
                 type = factor(c("dog", "cat", "mouse", "dog"), levels = c("dog", "cat", "mouse")),
                 age = c(3, 5, 2, 7));


### Use numeric indexation to get the value of column 'name' in the second row.

df[2, 1];


### Use logical indexation to get the rows where 'type' is equal to dog. Do not create
### the logical index manually, it should be the output of a logical operator (>, <, ==, !=, etc.)

df[df[, colnames(df) == "type"] == "dog",];


### Use indexation by name to get the age of Jerry. You can select the row of 'df' 
### corresponding to Jerry using any indexation method, but the age columns must be 
### selected using indexation by name.

df[3, "age"];


### Use indexation by variable ($ operator) in combination with logical indexation
### to get the name of the animals with age less than five.

df[df$age < 5, ]$name;



### (*) EXERCISE 4: A new prediction by Nostradamus has been recently discovered stating: 
### 'A machine learning algorithm will rule the world by 2018-11-08 16:50:00'.
### The date and time of this forecast is stored in the following variable

singularity <- "8AI50...18->000:16,11";


### You want to know how many minutes are left until this machine learning god
### comes to life. Write the R commands required to get this number.

singularity <- strptime(singularity, format = "%eAI%M...%y->%S0:%H,%m");
artificial_god <- difftime(singularity, Sys.time(), units = "mins");
as.numeric(artificial_god);



### (*) EXERCISE 5: You have the following dummy machine learning model:

model <- glm(Species ~ ., data = iris, family = "binomial");
model

### According to what we saw in class, which data type do you think is most suited to store 'model'
### (for instance, think about which R data type allows you to recover the original value of
### 'model' using the $ operator): a matrix, a data.table or a list? (QUESTION 3). 
### Create a variable of the data type you selected in the previous question containing 'model'. 

a <- data.table(c1=model)
a
a$c1
class(a)
class(a$c1)

l <- list(model = model);
l$model;
class(l)
### ANSWER TO 3: List.



### (**) EXERCISE 6: Convert the mtcars dataset (preloaded in R) into a data.table variable named 'dt'.
### Then count the number of cars (rows) with 6 cylinders using only one command with the format 
### dt[i, j, by] (maybe you do not need to use all of the three dimensions or indexes).

### EXPECTED OUTPUT (the name of the columns can be different in your output, but not their value):
#    n
# 1: 7

help(mtcars);
dt <- as.data.table(mtcars)
head(dt,2)
dt[ cyl ==6, list(n = .N)]


dt <- as.data.table(mtcars);
dt[cyl == 6, list(n = .N)];


### (**) EXERCISE 7: Using again 'dt', compute the maximum weight for each value of gear.
### Use only one command with the format dt[i, j, by] (maybe you do not need to use all of the three
### dimensions or indexes).

### EXPECTED OUTPUT (the name of the columns can be different in your output, but not their value):
#      gear max_w
# 1:    4   3.440
# 2:    3   5.424
# 3:    5   3.570

dt[,list(max_w = max(wt)),by = "gear" ]

dt[, list(max_w = max(wt)), by = "gear"];



### (**) EXERCISE 8: Using only one command with the format dt[i, j, by] (maybe you do not need to use 
### all of the three dimensions or indexes) compute, for each possible number of cylinders,
### the number of cars with automatic transmission and the number of cars with manual
### transmission. Take into account only cars with more than 3 gears.

### EXPECTED OUTPUT (the name of the columns can be different in your output, but not their value):
#     cyl    automatic  manual
# 1:   6         2        3
# 2:   4         2        8
# 3:   8         0        2

dt
head(dt,2)
View(dt)
help(mtcars)
dt[ gear > 3, list(automatic= sum(am == 0), manual = sum(am ==1) ) , by = "cyl"]



dt[gear > 3, list(automatic = sum(am == 0),
                  manual = sum(am == 1)), by = "cyl"];



### (***) EXERCISE 9: You have the following text corresponding to the 
### "Don Quijote de la Mancha" book.

book <- "In a village of La Mancha, the name of which I have no desire to call to mind, 
there lived not long since one of those gentlemen that keep a lance in the lance-rack, 
an old buckler, a lean hack, and a greyhound for coursing. An olla of rather more beef 
than mutton, a salad on most nights, scraps on Saturdays, lentils on Fridays, and a pigeon
or so extra on Sundays, made away with three-quarters of his income. The rest of it went 
in a doublet of fine cloth and velvet breeches and shoes to match for holidays, while on 
week-days he made a brave figure in his best homespun. He had in his house a housekeeper 
past forty, a niece under twenty, and a lad for the field and market-place, who used to 
saddle the hack as well as handle the bill-hook. The age of this gentleman of ours was 
bordering on fifty; he was of a hardy habit, spare, gaunt-featured, a very early riser 
and a great sportsman. They will have it his surname was Quixada or Quesada (for here 
there is some difference of opinion among the authors who write on the subject), although
from reasonable conjectures it seems plain that he was called Quexana. This, however, is 
of but little importance to our tale; it will be enough not to stray a hair's breadth from
the truth in the telling of it."

### Replace "---FILL---" in the following R command to convert this
### character variable into a vector with one word per position.

# words <- unlist(strsplit("---FILL---"));

words <- unlist(strsplit(book, " "));

### Now get the top 10 more frequent words

top_words <- names(sort(table(words), decreasing = TRUE)[1:10])

### Filter 'words' so only the values contained in these top 10 remain

words <- words[words %in% top_words];

### Finally, cast this character vector into a factor and plot a barplot with the
### top words frequencies

words <- as.factor(words);
plot(words);


### (***) EXERCISE 10: Read the R object "userbase.RData" provided with this exam. It represents a 
### database of clients of an airline, where each row corresponds to a flight purchase. Get the top 5
### users in terms of number of flights purchased via online channels. Take into account only flights
### bought after "2018-11-01". Get also the top 5 in terms of price per purchase.


# Read the R object "userbase.RData"
dat <-readRDS("/Users/amand/Documents/IE/Programming R/Quizz2/userbase.RData");
dat

# Take into account only flights bought after "2018-11-01"
head(dat)
dat[bookind_date>=as.Date("2018-11-01")];

dat <- dat[fecha_reserva >= as.Date("2018-11-01")];

# Get the top 5 users in terms of number of flights purchased via online channels

head(dat,2)
dat[sale_channel=="online",list(n.purchases = .N),by = "user"][1:5,]
head(dat[sale_channel=="online",list(n_purchases= .N), by = "user"][order(n_purchases,
                                                                          decreasing = TRUE)],5);
head(dat[canal_venta == "online", list(n_purchases = .N), by = "user"][order(n_purchases,
                                                                             decreasing = TRUE)], 5);

# Get also the top 5 in terms of price per purchase.
head(dat[canal_venta == "online", list(avg_price = mean(price)), by = "user"][order(avg_price,
                                                                                    decreasing = TRUE)], 5)
;

library(data.table);


#########################################################################################################################

#### (*) Exercise 1: You have the following vector and index ####

v <- runif(100, 1, 10);
index <- floor(runif(1,1,100));

#### Create an if clause so the value of 'v' in the position indicated by
#### 'index' is only printed if this value is greater than 5.

if (v[index] > 5){
  print(v[index]);
}

#### (*) Exercise 2: Complete now the following if- else if -else clause in order ####
#### to make it print your name when 'v[index]' is between 2 and 5 (both) included.

if (v[index] > 5){
  print(v[index]);
} else if (v[index] >= 2 & v[index] <= 5){
  print("Jesus Prada Alonso.")
} else {
  print("Default clause.")
}



#### (*) Exercise 3: Create a for loop that go through all the values of 'v'####
#### and print them if the value is greater than five.

for (index in 1:length(v)){
  if (v[index] > 5){
    print(v[index]);
  }
}

#### (*) Exercise 4: Perform the same operation as in Exercise 3 but this time ####
#### using a while clause and also replicate the operation using repeat clause + break

## WHILE
index <- 1;

while (index <= length(v)){
  if (v[index] > 5){
    print(v[index]);
  }
  index <- index + 1;
}

## REPEAT + BREAK
index <- 1;

repeat{
  if (v[index] > 5){
    print(v[index]);
  }
  index <- index + 1;
  if (index > length(v)){break;}
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

#### (**) Exercise 6: Use the NEXT command together with a loop of your choice ####
#### (for, while or repeat) to print only the values of 'v' in positions that
#### correspond to even numbers.

for (i in 1:length(v)){
  if (i %% 2 != 0){next;}
  print(sprintf("Position %s: %f", i, v[i]));
}

#### (**) Exercise 7: You have the following matrix ####

M <- matrix(runif(10^4, min = 1, max = 10), ncol = 6);
dim(M);
head(M);

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

#### (**) Exercise 8: You have the following dataset (change the dimension if ####
#### you have memory problems)

dimension <- 10^8;
df <- as.data.frame(matrix(sample(c(FALSE,TRUE), dimension, replace = TRUE),
                           ncol = 100));
colnames(df) <- paste0("col_", 1:100);
dim(df);
head(df[,1:8]);

#### and this function (it is not really relevant what the function actually does)
my_fun <- function(x){
  perc_positives <-  sum(x == TRUE)/length(x);
  if (perc_positives > 0.5){
    ret <- "positive";
  } else {
    ret <- "negative";
  }
  return(ret);
}

#### Compute the given function 'my_fun' for each column of this given
#### dataset and store the output in a variable of your choice.
#### Do it using a for loop, apply, lapply and sapply. Which option is fastest?
#### To which class belong the output of each operation? Suggestion: system.time()

## FOR
t1 <- system.time({
values_1 <- c();
for (col in colnames(df)){
  values_1 <- c(values_1, my_fun(df[, col]));
}
})[3];
names(values_1) <- colnames(df);
print(values_1);

## APPLY
t2 <- system.time({
  values_2 <- apply(df, 2, my_fun)
  })[3];
print(values_2);

## LAPPLY
t3 <- system.time({
  values_3 <- lapply(df, my_fun)
  })[3];
print(values_3);

## SAPPLY
t4 <- system.time({
  values_4 <- sapply(df, my_fun)
  })[3];
print(values_4);


## TIME COMPARISON
print(sprintf("For: %f - apply: %f - lapply: %f - sapply: %f",
              t1, t2, t3, t4));

## CLASS COMPARISON
print(sprintf("For: %s - apply: %s - lapply: %s - sapply: %s",
              class(values_1),class(values_2),
              class(values_3), class(values_4)));



#### (***) Exercise 9: Let's compute my_fun only for the first five ####
#### columns using a for loop

t5 <- system.time({
  values_5 <- c();
  for (col in colnames(df)[1:5]){
    values_5 <- c(values_5, my_fun(df[, col]));
  }
})[3];
names(values_5) <- colnames(df)[1:5];
print(values_5);


### Now cast df to a data.table

dt <- as.data.table(df);


#### Using a dt[i, j, by] compute the same operation as the previous for loop.
#### Compare again computation time and class obtained

## dt[i, j, by]

t6 <-  system.time({
  values_6 <- dt[, list(res_1 = my_fun(col_1),
                        res_2= my_fun(col_2),
                        res_3= my_fun(col_3),
                        res_4= my_fun(col_4),
                        res_5= my_fun(col_5))]
  })[3];
values_6;

## TIME COMPARISON
print(sprintf("For: %f - dt[i, j, by]: %f",
              t5, t6));

## CLASS COMPARISON
print(sprintf("For: %s - dt[i, j, by]: %s",
              class(values_5)[1],class(values_6)[1]));



#### (***) Exercise 10: Install and load packages foreach and doParallel. ####

install.packages("foreach");
install.packages("doParallel");
library(foreach);
library(doParallel);


#### Read the documentation of each package

help(foreach);
help(doParallel);


#### Create a cluster of several cores. Suggestion: registerDoParallel()

registerDoParallel(cores = 2);

#### A foreach loop has the following syntax
for (iterator in 1:10){
  iterator;
}

result <- foreach (iterator = 1:10)%dopar%{
  iterator;
};
print(result);

#### Convert the for loop in exercise 8 into a foreach loop to parallelize
#### the computation. Compare times and classes obtained between these two loops.

t7 <- system.time({
  values_7 <- foreach (col = colnames(df))%dopar%{
    my_fun(df[, col]);
  }})[3];
print(values_7);

## TIME COMPARISON
print(sprintf("For: %f - foreach: %f",
              t1, t7));

## CLASS COMPARISON
print(sprintf("For: %s - foreach: %s",
              class(values_1),class(values_7)));


library(data.table);

#########################################################################################################################

#### (*) Exercise 1: Together with this .R file you received two separate .csv files. ####
#### Read them and store their content in a R variable both using read.table (or read.csv)
#### and fread in combination with file.path.

library(data.table);
root_path <- "/Users/Falendor/Dropbox/Planes/proyectos/academia/IE/sessions/14-15/exercises/";

f1_a <- read.table(file.path(root_path, "file.csv"));
f1_b <- fread(file.path(root_path, "file.csv"));

f2_a <- read.table(file.path(root_path, "input.csv"), header = TRUE, sep = "|");
f2_b <- fread(file.path(root_path, "input.csv"));


#### (*) Exercise 2: Create a function to compute the frequency (in percentage) ####
#### of the mode or most frequent value in a vector. Call this function for each
#### column in 'df'

df <- data.frame(ID = sample(1:100, 1000, replace = TRUE),
                 animal = sample(c("cat", "dog", "mouse"), 1000, replace = TRUE),
                 abandoned = sample(c(TRUE, FALSE), 1000, replace = TRUE));

#### The output should be

# ID animal  alive 
# 2.1   34.5   52.3 

my_mode <- function(x) {
    x <- as.factor(x);
    ret <- names(sort(table(x), decreasing = TRUE))[1];
    
    return(ret);
}

main_value_ratio <- function(x){
    x <- as.character(x);
    return(100*sum(x == my_mode(x), na.rm = TRUE)/length(x));
}

sapply(df, main_value_ratio);



#### (*) Exercise 3: Create a function to replace all missing values ####
#### in a vector for its median. Then call this function for each column of 'df'

df <- data.frame(Var1 = sample(c(1:100, NA), 1000, replace = TRUE),
                 Var2 = sample(c(1:100, NA), 1000, replace = TRUE),
                 Var3 = sample(c(1:100, NA), 1000, replace = TRUE),
                 Var4 = sample(c(1:100, NA), 1000, replace = TRUE),
                 Var5 = sample(c(1:100, NA), 1000, replace = TRUE));

fill_missing_with_median <- function(x){
    x <- as.numeric(x);
    x[is.na(x)] <- median(x, na.rm = TRUE);
    return(x);
}

sapply(df, function(x){sum(is.na(x))});
df <- as.data.frame(sapply(df, fill_missing_with_median));
sapply(df, function(x){sum(is.na(x))});

#### (*) Exercise 4: Create a function to tipify a vector, i.e substract ####
#### the mean and divide by the standard deviation. As usual call this function
#### for each variable of 'df', where 'df' is the same as in the previous exercise.

tipify <- function(x){
    mu <- mean(x, na.rm = TRUE);
    s <- sd(x, na.rm = TRUE);
    
    # Special condition for constant variables
    s[s == 0] <- 1;
    
    # Tipify
    x <- (x - mu) / s;
}

sapply(df, mean, na.rm = TRUE);
sapply(df, sd, na.rm = TRUE);

df <- as.data.frame(sapply(df, tipify));

sapply(df, mean, na.rm = TRUE);
sapply(df, sd, na.rm = TRUE);


#### (*) Exercise 5: Use RStudio debugging options to find the error ####
#### in this code that makes the loop to go on forever. Bear in mind that 
#### the function called here can be found in 'aux_functions.R', a file provided 
#### together with this script. Suggestion: Save everything before running
#### the infinite loop!

index <- 1;
counter <- 1;

while(index != 2){
    counter <- debug_f(counter);
    index <- index + 1;
}


#### (**) Exercise 6: Categorical variables like this one ####

v <- c("cat", "dog", "mouse", "dog", "cat");

#### cannot be used in most machine learning models. To use this
#### type of information when applying these models usually one-hot
#### encoding (https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
#### is performed. This means to transform 'v' into three variables like this

#     cat dog mouse
# 1:   1   0     0
# 2:   0   1     0
# 3:   0   0     1
# 4:   0   1     0
# 5:   1   0     0

#### Note: Actually, the last column is redundant. Do you know why? 

#### Create a function 'onehot' to perform one-hot encoding over a vector
#### and call this function on the following vector 'v'

v <- sample(c("cat", "dog", "mouse"), 1000, replace = TRUE);


onehot <- function(x){
    values<-unique(x);
    
    ret<-data.table(matrix(0,nrow = length(x),ncol=length(values)))
    
    for (i in 1:length(values)){
        ret[, i] <- as.numeric(x==values[i]);
    }
    
    colnames(ret)<-values;
    return(ret);
}


onehot_v <- onehot(v);

table(v);
sum(onehot_v$cat);
sum(onehot_v$dog);
sum(onehot_v$mouse);

### Equivalent option
onehot <- function(x){
    values <- unique(x)[-length(unique(x))];
    
    ret<-data.table(matrix(0,nrow = length(x),ncol=length(values)))
    
    for (i in 1:length(values)){
        ret[, i] <- as.numeric(x==values[i])
    }
    
    colnames(ret)<-values;
    return(ret);
}


onehot_v <- onehot(v);

table(v);
sum(onehot_v$cat);
sum(onehot_v$dog);
sum(onehot_v$mouse);

#### (**) Exercise 7: One type of plot that is particularly interesting when ####
#### performing exploratory data analysis is the density function plot. Try
#### to replicate the plot shown in 'plot.png' using the column price of
#### the dasaset contained in "real_userbase.RData" (this file is given together  
#### with this set of exercises). Suggestion: Use density() after replacing missing
#### values.

# Read file
dat <- readRDS(file.path(root_path, "real_userbase.RData"));

# Plot density
plot(density(dat$price), col = "blue"); # Error! Why?

# Replace missing values
sapply(dat, function(x){sum(is.na(x))});
dat$price[is.na(dat$price)] <- mean(dat$price, na.rm = TRUE);
sapply(dat, function(x){sum(is.na(x))});

# Plot density
plot(density(dat$price), col = "blue");

#### (**) Exercise 8: One usual pre-processing step before applying a machine ####
#### learning model is to remove "redundant" and "irrelevant" variables. Let's
#### define a redundant variable as one with a correlation greater (in absolute value) 
#### than 0.9 to any other variable of the dataset, and irrelevant as one with less than 
#### 0.5 (in absolute value) correlation with the target you want to predict. 
#### Detect redundant and irrelevant variables from mtcars using the am column as target.

# REDUNDANT VARIABLES
remove_redundant <- function(correlations,redundant_threshold){
  redundancy<-apply(correlations,2,function(x){which(x>redundant_threshold)});
  redundancy<-redundancy[which(sapply(redundancy,length)>1)]
  
  redundant_variables<-c();
  for (i in redundancy){
    imp<-sort(correlations[1,i],decreasing = TRUE);
    redundant_variables<-c(redundant_variables,names(imp)[2:length(i)])
  }
  redundant_variables<-unique(redundant_variables);
  return(redundant_variables);
} 

# IRRELEVANT VARIABLES
remove_irrelevant<-function(correlations,irrelevant_threshold, target){
  index <- which(target == colnames(correlations));
  
  # Irrelevant variables
  relevance<-correlations[index,-index];
  irrelevant_variables<-names(relevance)[is.na(relevance) | relevance<irrelevant_threshold];
  return(irrelevant_variables);
}


cors <- abs(cor(mtcars));
target <- "am";
remove_redundant(cors, 0.9);
data.table(mtcars)[, -redundant_vars, with= F]
remove_irrelevant(cors, 0.5, target);

#### What happens if we define the redundant threshold as 0.99 and redundancy as
#### lower than 0.1? 

remove_redundant(cors, 0.99);
remove_irrelevant(cors, 0.001, target);



#### (***) Exercise 9: Use SMOTE to perform oversampling of mtcars dataset ####
#### to get a total of 39 rows with am = 0 and 52 rows with am = 1. Suggestion:
#### You must correctly set perc.over and perc.under arguments.

# Load library
library(DMwR);

# Cast target to a factor
mtcars$am <- as.factor(mtcars$am);

# Perform SMOTE and compare dimensions
table(mtcars$am);
smote_mtcars <- SMOTE(am ~ ., mtcars, perc.over = 100, perc.under = 100, k=1);
table(smote_mtcars$am);


#### (***) Exercise 10: Dimensionality reduction is commonly performed ####
#### in machine learning projects for computational and/or model accuracy optimization.
#### Use filterVarImp() of caret library to select the top 5 most relevant
#### variables in mtcars when am is the target you want to predict

library(caret);

select_important<-function(dat, n_vars, y){
  varimp <- filterVarImp(x = dat, y=y, nonpara=TRUE);
  varimp <- data.table(variable=rownames(varimp),imp=varimp[, 1]);
  varimp <- varimp[order(-imp)];
  selected <- varimp$variable[1:n_vars];
  return(selected);
}

setDT(mtcars);
select_important(dat = mtcars[, -"am"], n_vars = 3, y = mtcars$am);


library(data.table);
set.seed(140);

#########################################################################################################################

#### (*) Exercise 1: Mean Absolute Percentage Error, MAPE, and Root Square Mean Error, RMSE are two ####
#### popular metrics when measuring regression models. The first one is defined as:

#### https://en.wikipedia.org/wiki/Mean_absolute_percentage_error


#### and the second one as:

#### https://en.wikipedia.org/wiki/Root-mean-square_deviation

#### Using these definitions create two functions called MAPE and RMSE, one to compute each metric, and 
#### call them using these vectors reprenting predictions and their corresponding real values.

real <- sample(1:10, 100, replace = TRUE);
predictions <- real + runif(length(real), -1, 1);


MAPE <- function(prediction, real){
  ret <- 100*mean(abs((prediction - real) / real));
  return(ret);
}

RMSE <- function(prediction, real){
  mse <- mean((prediction - real)^2)
  ret <- sqrt(mse);
  return(ret);
}

MAPE(predictions, real);
RMSE(predictions, real);

#### (*) Exercise 2: In the script we saw in class we repeated three steps (independent of the model) ####
#### several times for our regression models:

#### 1) Get model predictions
#### 2) Get errors
#### 3) Compute Metrics (MSE and MAE)

#### Imagine that you have the following model and dataset

index_train <- sample(1:nrow(mtcars), 0.7*nrow(mtcars));
model <- lm(mpg ~ ., data = mtcars[index_train,]);
dataset <- setDT(mtcars[-index_train,]);
dataset;

#### Create a single function 'ml_wrapper' to compute the three previous steps with a call like this one

ml_wrapper(model = model, dataset = dataset, target = dataset$mpg);

#### The output should be

# $`predictions`
# 1        2        3        4        5        6        7        8        9       10 
# 23.66050 19.37174 17.59102 19.26578 17.03017 14.09378 13.84086 23.68908 23.58032 26.01283 
# 
# $errors
# 1         2         3         4         5         6         7         8         9        10 
# 2.660501 -2.028258 -1.108979 -5.134222  1.830166  3.693779  3.440856 -6.710920  7.780324  4.612833 
# 
# $mse
# [1] 19.45
# 
# $mae
# [1] 3.9


ml_wrapper <- function(model, dataset, target){
  
  # 1) Get model predictions
  predictions <- predict(model, newdata = dataset);
  
  # 2) Get errors
  errors <- predictions - target;
  
  # 3) Compute Metrics (MSE and MAE)
  mse <- round(mean(errors^2), 2);
  mae <- round(mean(abs(errors)), 2);
  
  return(list(predictions = predictions,
         errors = errors,
         mse = mse,
         mae = mae));
}

#### (*) Exercise 3: These two commands are equivalent ####

model_1 <- lm(mpg ~ wt, data = mtcars[index_train,]);
model_2 <- lm("mpg ~ wt", data = mtcars[index_train,]);
identical(model_1$coefficients, model_2$coefficients);

#### Following the syntax of model_2, try now to create an equivalent call to the following one

model_1 <- lm(mpg ~ cyl + disp + hp + drat + wt + qsec + vs + am + gear, data = mtcars[index_train,]);

#### without explicitly writing the names of the predictor variables. Take notice that the only variables
#### not used as predictors are "carb" and the target itself, "mpg". Suggestion: Use paste0.

formula <- paste0("mpg ~ ",paste0(setdiff(colnames(mtcars), c("mpg", "carb")), collapse = " + "));
model_2 <- lm(formula, data = mtcars[index_train,]);
identical(model_1$coefficients, model_2$coefficients);


#### (*) Exercise 4: Imagine you have the following metrics evaluating several regression models ####
#### for the same problem. 

comp <- data.table(model = c("model1", "model2", "model3", "model4", "model5"), 
                   mae_train = c(0.8, 0.3, 0.6, 1.7, 3.1),
                   mae_test = c(1.0, 2.8, 0.7, 1.7, 5.6));
comp;

#### For each model, try to guess if it that is a case of overfitting, underfitting or none. 

# model 1: None
# model 2: Overfitting
# model 3: None
# model 4: Underfitting
# model 5: Underfitting

#### (*) Exercise 5: Imagine you have the following machine learning model ####

library(e1071);

index_train <- sample(1:nrow(iris), 0.7*nrow(iris));
train <- iris[index_train,];
model <- svm(Species ~ ., data = train, kernel="radial"); 
model;

#### Is this a classification or a regression problem? Try to build a linear model (regressor or
#### classifier depending on your answer) for this problem. Suggestion: [2.1.3] section of script_s16-17.R

model <- glm(Species ~ ., data = train, family = "binomial");


#### (**) Exercise 6: Considering the problem in exercise 5, let's use accuracy as ####
#### evaluation metric for the following model

model <- svm(Species ~ ., data = train, kernel="radial",
             cost = 10^-3, gamma = 10^-5); 

accuracy <- function(prediction, real){
  ret <- 100*sum(prediction == real)/length(real);
  return(ret);
}

ml_wrapper <- function(model, dataset, target, metric_f){
  
  # 1) Get model predictions
  predictions <- predict(model, newdata = dataset);
  
  # 2) Compute Metrics
  metric <- metric_f(predictions, target);
  
  return(metric);
}

results_train <- ml_wrapper(model, iris[index_train,], iris$Species[index_train], accuracy);
results_test <- ml_wrapper(model, iris[-index_train,], iris$Species[-index_train], accuracy);
results_train;
results_test;

#### This model has a clear problem of underfitting. Can you fix this problem without using
#### a hyperparameter grid search? Suggestion: low values of cost means low complexity.

model <- svm(Species ~ ., data = train, kernel="radial",
             cost = 1); 

results_train <- ml_wrapper(model, iris[index_train,], iris$Species[index_train], accuracy);
results_test <- ml_wrapper(model, iris[-index_train,], iris$Species[-index_train], accuracy);
results_train;
results_test;



#### (**) Exercise 7: Random Forests are another popular choice for regression models ####
#### You can install, load and check the documentation of its library like this

# install.packages("randomForest")
library(randomForest);
help(randomForest);

#### Build a Random Forest model to predict 'Species' in the iris dataset and
#### compare its results with the ones you achieved in exercise 7.

model <- randomForest(Species ~ ., data = train); 

results_train <- ml_wrapper(model, iris[index_train,], iris$Species[index_train], accuracy);
results_test <- ml_wrapper(model, iris[-index_train,], iris$Species[-index_train], accuracy);
results_train;
results_test;

#### (**) Exercise 8: The use of global accuracy as evaluation metric can have ####
#### significant drawbacks. For instance, imagine you are working in an unbalance
#### problem like this one

target <- c(rep(1,990),rep(0,10));
table(target);

#### if we build a dummy model that simply outputs one regardless of the input

predictions <- rep(1, length(target));

#### we would get an accuracy of 99%

accuracy(predictions, target);


#### Do you think a dummy constant model that always give the value of 1 as prediction
#### is a good machine learning model? That's why the Area Under the Curve, AUC, metric
#### is usually the choice in these situations. You can see it's definition here:
#### https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=es-419
#### Try to create a function to compute this AUC value. For instance, the following calls

AUC(target, predictions);
AUC(target, runif(length(target), 0, 1));
AUC(target, target + runif(length(target), 0, 2));
AUC(target, target);

#### Should give you the following results respectively
# Area under the curve: 0.5
# Area under the curve: 0.5075
# Area under the curve: 0.8555
# Area under the curve: 1

#### Suggestion: use pROC library

# install.packages("pROC");
library(pROC);
AUC <- function(target, prediction){
  roc_curve <- roc(target, prediction);
  auc(roc_curve);
}

#### (***) Exercise 9: Replicate the grid search we did in [1.2.3] of Session 16-17 ####
#### class material but using now foreach. The number of cores selected to do the parallelization
#### is up to you.

# setting seed to reproduce results of random sampling
set.seed(100); 

# Convert mtcars to data.table
dat <- as.data.table(mtcars);

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

# Load libraries
library(doParallel);
library(foreach);

# Start cluster
stopImplicitCluster();
registerDoParallel(cores = detectCores());

### Define grid
c_values <- 10^seq(from = -3, to = 3, by = 1);
eps_values <- 10^seq(from = -3, to = 0, by = 1);
gamma_values <- 10^seq(from = -3, to = 3, by = 1);

### Compute grid search
grid_results <-  foreach (c = c_values, .combine = rbind)%:%
  foreach (eps = eps_values, .combine = rbind)%:%
    foreach (gamma = gamma_values, .combine = rbind)%dopar%{
      
      library(e1071);
      library(data.table);
      
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
      data.table(c = c, eps = eps, gamma = gamma, 
                 mse_train = mse_train, mae_train = mae_train,
                 mse_val = mse_val, mae_val = mae_val);
    }

# Order results by increasing mse and mae
grid_results <- grid_results[order(mse_val, mae_val)];

# Check results
grid_results[1];


#### You can also use random search so you do not try all possible combinations of
#### points. An heuristic will decide which points to check. You can do 
#### that using caret library: https://topepo.github.io/caret/random-hyperparameter-search.html

library(caret);

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           search = "random");

fit <- train(mpg ~ ., data = train, 
                 method = "svmRadialSigma",
                 metric = "RMSE",
                 tuneLength = 30,
                 trControl = fitControl,
                 tuneGrid = data.frame(sigma = gamma_values,
                                       C = c_values));

fit;
grid_results <- data.table(fit$results)[order(RMSE)];
grid_results[1];

#### (***) Exercise 10: Your goal now is to predict the 'wt' column of mtcars. ####
#### Find a model with less than 0.05 of MSE for the test set.

# Start cluster
stopImplicitCluster();
registerDoParallel(cores = detectCores());

### Define grid
c_values <- seq(from = 200, to = 300, length.out = 10);
eps_values <- seq(from = 0.04, to = 0.07, length.out = 10);
gamma_values <- seq(from = 0.006, to = 0.01, length.out = 10);

### Compute grid search
grid_results <-  foreach (c = c_values, .combine = rbind)%:%
  foreach (eps = eps_values, .combine = rbind)%:%
  foreach (gamma = gamma_values, .combine = rbind)%dopar%{
    library(e1071);
    library(data.table);
    
    print(sprintf("Start of c = %s - eps = %s - gamma = %s", c, eps, gamma));
    
    # train SVM model with a particular set of hyperparamets
    model <- svm(wt ~ ., data = train, kernel="radial",
                 cost = c, epsilon = eps, gamma = gamma);
    
    # Get model predictions
    predictions_train <- predict(model, newdata = train);
    predictions_val <- predict(model, newdata = val);
    
    # Get errors
    errors_train <- predictions_train - train$wt;
    errors_val <- predictions_val - val$wt;
    
    # Compute Metrics
    mse_train <- round(mean(errors_train^2), 2);
    mse_val <- round(mean(errors_val^2), 2);
    
    # Build comparison table
    data.table(c = c, eps = eps, gamma = gamma, 
               mse_train = mse_train,
               mse_val = mse_val);
  }

# Order results by increasing mse and mae
grid_results <- grid_results[order(mse_val, mse_train)];

# Check results
best <- grid_results[1];

### Train final model
# train SVM model with best found set of hyperparamets
model <- svm(wt ~ ., data = train, kernel="radial",
             cost = best$c, epsilon = best$eps, gamma = best$gamma);

# Get model predictions
predictions_train <- predict(model, newdata = train);
predictions_val <- predict(model, newdata = val);
predictions_test <- predict(model, newdata = test);

# Get errors
errors_train <- predictions_train - train$wt;
errors_val <- predictions_val - val$wt;
errors_test <- predictions_test - test$wt;

# Compute Metrics
mse_train <- round(mean(errors_train^2), 2);
mse_val <- round(mean(errors_val^2), 2);
mse_test <- round(mean(errors_test^2), 2);

## Summary
sprintf("MSE_train = %s - MSE_val = %s - MSE_test = %s", mse_train, mse_val, mse_test);

#########################################################################################################################

#### (*) Exercise 1: Create a function, called 'my_and', that returns as output  ####
#### TRUE only if the first argument, named 'x', and the second argument, named 'y', 
#### are both TRUE. Use default values so that the output of

my_and();


#### is TRUE

my_and <- function(x = TRUE, y = TRUE) {
  result <- x & y;
  return(result);
}



#### (*) Exercise 2: You have the following vector ####

v <- sample(c(TRUE, FALSE), 100, replace = TRUE);


#### Create a loop that iterates through all the positions of 'v' and, for each ####
#### iteration, prints "TRUE value found" if the value of 'v' in that position
#### is TRUE. Do this using both a for and a while loop. Each loop should contain an if
#### (or if-else) clause.

for (index in 1:length(v)){
  if (v[index]){
    print("TRUE value found")
  } else {
  }
}

index <- 1;
while (index <= length(v)){
  if (v[index]){
    print("TRUE value found")
  } else {
  }
  index <- index + 1;
}


#### (*) Exercise 3: For this exercise you will be using the same vector 'v' as in exercise 2. ####
#### Your objective now is to create a loop (you can choose which type) that iterates through the positions
#### of 'v' until it finds an index where 'v[index]' is equal to 'v[index+1]', in that moment you should
#### print the value of 'v[index]' two times and stop the loop using the "break" command. Beware infinite loops.


for (index in 1:length(v)){
  if (v[index] == v[index+1]){
    print(v[index]);
    print(v[index]);
    break;
  } 
}

#### (*) Exercise 4: Source the file "auxiliary_functions.R" that comes ####
#### with this test. Compute the function inside this script for each column 
#### of the following dummy dataset and store the result in a variable. 

df <- data.frame(v1 =  sample(letters, 100, replace = TRUE),
                  v2 = sample(letters, 100, replace = TRUE),
                  v3 = sample(letters, 100, replace = TRUE));


#### Do this using both a while loop and apply(). To which class belongs the
#### resulting variable in each case? (QUESTION 1)

source('C:/Users/Falendor/Dropbox/Planes/proyectos/academia/IE/sessions/13b/auxiliary_functions.R');

ret_1 <- c();
n_col <- 1;
while (n_col <= ncol(df)){
  ret_1 <- c(ret_1, my_mode(df[, n_col]));
  n_col <- n_col + 1;
}
ret_1;

ret_2 <- apply(df, 2, my_mode);

class(ret_1);
class(ret_2);

#### ANSWER 1: character for both methods.

#### (*) Exercise 5: You have the following vector ####

v <- sample(0:100, 100, replace = TRUE);

#### Try to replicate the plot given in quiz2_plot.png together with this quiz using
#### basic plot functions (not ggplot). Bear in mind that the actual values plotted are going
#### to be probably different in your plot as 'v' is a random vector.

hist(v, col = "purple", fg = "red",
     main = "TITLE", sub = "subtitle", xlab = "x values", 
     ylim = c(-10,30),
     font = 4, las = 2, cex.axis = 0.5,
     breaks = 5);


#### (**) Exercise 6: Create a function called 'count_animals' which takes advantage of  ####
#### the ... argument functionality to return a list with two variables: the total number of "cat" 
#### words that are passed to the function when called,  stored in a variable called 'n_cats', 
#### and the total number of "dog", stored in a variable called 'n_dogs'. You can assume that this
#### function would be called only with character or factor variables. For instance, the 
#### following call

count_animals(v1 = c("cat"), 
             v2 = letters[1:4], 
             v3 = c("1", "2"),
             v4 = c("dog"),
             v5 = factor(c("cat", "dog", "dog"), levels = c("cat", "dog")));

#### should give this output

# $`n_cats`
# [1] 2
# 
# $n_dogs
# [1] 3

count_animals <- function(...){
  arguments <- list(...);
  n_cats <- 0;
  n_dogs <- 0;
  for (var in arguments){
    n_cats <- n_cats + sum(var == "cat");
    n_dogs <- n_dogs + sum(var == "dog");
  }
  return(list(n_cats = n_cats, n_dogs = n_dogs));
}


#### (**) Exercise 7: Try to replicate the plot given in quiz2_plot_2.png together  #### 
#### with this quiz using ggplot functions. Bear in mind that the values plotted come 
#### from the mtcars dataset.

### Create plot
my_plot <- ggplot(as.data.table(mtcars), 
                  aes(x=1:nrow(mtcars), y = cyl));

### Add layer of points
my_plot <- my_plot + geom_point(col="purple");


### Add layer to set main and axis titles
my_plot <- my_plot + 
  labs(subtitle="From mtcars dataset", 
       x= "index", y="cyl", caption="mtcars dataset cyl");

my_plot <- my_plot +  
  scale_x_continuous(breaks=seq(0, nrow(mtcars), 3),
                     labels = paste0("v_",seq(0, nrow(mtcars), 3)));


### Print plot
print(my_plot);



#### (**) Exercise 8: Open the R Markdown 'quiz2_rmarkdown.Rmd' given with this file.  ####
#### Make the following modifications to this R Markdown

#### 1) Modify the header to include a table of contents.
#### 2) Modify the general configuration chunk so for any R code both code and
#### output are not included by default.
#### 3) Create a new text block at the bottom with a header of a bigger size than the one
#### already present in the R Markdown.
#### 4) Create a new code block below the previous text block to compute
#### the mean value of the cyl column of the mtcars dataset. The code of this block
#### must not appear in the output document but its result should.


#### You have to submit the modified 'quiz2_rmarkdown.Rmd' together with this .R file.


#### (***) Exercise 9: Modify the function in auxiliary_functions.R so now it returns ####
#### the less frequent value of x instead of the mode or most frequent value. Then use a 
#### dt[i, j, by] operation to call this function to compute the less frequent value of all 
#### the columns in mtcars and store the output in a variable called 'less_frequent'.
#### Use of .SD will imply a higher score. The output should look like this

#   mpg       cyl   disp     hp    drat     wt       qsec     vs     am       gear   carb 
#  "13.3"     "6"  "71.1"    "52"  "2.93" "1.513"  "14.5"     "1"     "1"     "5"     "6" 


my_mode <- function(x) {
  x <- as.factor(x);
  ret <- names(sort(table(x), decreasing = FALSE))[1];
  return(ret);
}

dt <- as.data.table(mtcars);

less_frequent <- dt[, sapply(.SD, my_mode)];
names(less_frequent) <- colnames(mtcars);

#### (***) Exercise 10: Open the Shiny app in folder 'quiz2_shiny'.  ####
#### Make the following modifications to the ui.R and server.R files

#### 1) Create a new sliderInput called 'filter' where an user can
#### choose a range of rows to filter mtcars.
#### 2) Use the value of this 'filter' input together with the already given 'columns'
#### input to print a DT table of mtcars for the columns selected in 'columns' and
#### the rows selected in 'filter'. Suggestion: You should probably use a command like
#### dt[i, j, by] command to make this work.
#### 3) If a user changes the values of 'columns' or filter' selected 
###  the DT table should also automatically change.
#### 4) You can print this DT table at any position in your shiny app.


#### You have to submit this 'quiz2_shiny' folder together with this .R file.

#########################################################################################################################

#### (*) Exercise 1: Create a function, called 'my_paste', that returns as output  ####
#### the result of concatenating its first argument, named 'x', to the second 
#### argument, named 'y', separated by a '_' symbol. Use default values so that 
### the output of

my_paste();


#### is "first_point"

my_paste <- function(x = "first", y = "point") {
  result <- paste(x, y, sep = "_")
  return(result);
}



#### (*) Exercise 2: You have the following vector ####

v <- sample(c("cat", "dog", "mouse", "rat"), 100, replace = TRUE);


#### Create a loop that iterates through all the positions of 'v' and, for each ####
#### iteration, prints "letter a found" if the value of 'v' in that position
#### contains the letter "a". Do this using both a for and a while loop. 
#### Each loop should contain an if (or if-else) clause.

for (index in 1:length(v)){
  if (grepl("a", v[index])){
    print("letter a found")
  } else {
  }
}

index <- 1;
while (index <= length(v)){
  if (grepl("a", v[index])){
    print("letter a found")
  } else {
  }
  index <- index + 1;
}


#### (*) Exercise 3: For this exercise you will be using the same vector 'v' as ####
#### in exercise 2 and this second vector

v2 <- sample(c("cat", "dog", "mouse", "rat"), 100, replace = TRUE);

#### Your objective now is to create a loop (you can choose which type) that iterates 
#### through the positions of 'v' until it finds an index where 'v[index]' is equal to 
#### 'v2[index]', in that moment you should print the value of 'index' and stop the loop
#### using the "break" command. Beware infinite loops.

for (index in 1:length(v)){
  if (v[index] == v2[index]){
    print(index);
    break;
  } 
}

#### (*) Exercise 4: Source the file "auxiliary_functions.R" that comes ####
#### with this test. Compute the function inside this script for each column 
#### of the following dummy dataset and store the result in a variable. 

df <- data.frame(v1 =  1:100,
                  v2 = 101:200,
                  v3 = 201:300);


#### Do this using both a while loop and lapply(). To which class belongs the
#### resulting variable in each case? (QUESTION 1)

source('C:/Users/Falendor/Dropbox/Planes/proyectos/academia/IE/sessions/13/auxiliary_functions.R')

ret_1 <- c();
n_col <- 1;
while (n_col <= ncol(df)){
  ret_1 <- c(ret_1, my_percentile(df[, n_col]));
  n_col <- n_col + 1;
}
ret_1;

ret_2 <- lapply(df, my_percentile);

class(ret_1);
class(ret_2);

#### ANSWER 1: numeric for the while loop and list for lapply().

#### (*) Exercise 5: You have the following vectors ####

v <- sample(0:100, 100, replace = TRUE);
v2 <- sample(10:20, 100, replace = TRUE);

#### Try to replicate the plot given in quiz2_plot.png together with this quiz using
#### basic plot functions (not ggplot). Bear in mind that the actual values plotted are 
#### going to be probably different in your plot as 'v' and 'v2' are random vectors.

plot(v, col = "blue", fg = "green", type = "l",
     sub = "title", ylab = "y", 
     xlim = c(-10,110), ylim = c(10,60),
     lty = 4, font = 2, las = 1, cex.axis = 0.5);
lines(v2, col = "red", type = "p");
legend("bottomright", legend=c("v", "v2"),
       col=c("blue", "red"), lty = c(4,1), cex=1);


#### (**) Exercise 6: Create a function called 'count_numeric' which takes advantage of  ####
#### the ... argument functionality to return a list with two variables: the total number of 
#### numeric variables that are passed to the function when called, stored in a variable called 
#### 'n_num', and their positions in the list of arguments, stored in a variable called 'indexes'. 
#### For instance, the following call

count_numeric(v1 = c(1,2,3,4), 
             v2 = letters[1:4], 
             v3 = 101:200,
             v4 = c(5, 6, 7),
             v5 = factor(c("cat", "dog", "dog"), levels = c("cat", "dog")));

#### should give this output

# $`n_num`
# [1] 2
# 
# $indexes
# [1] 1 4

count_numeric <- function(...){
  arguments <- list(...);
  n_num <- 0;
  indexes <- c();
  for (index in 1:length(arguments)){
    var <- arguments[[index]];
    if (class(var) == "numeric"){
      n_num <- n_num + 1;
      indexes <- c(indexes, index);
    }
  }
  return(list(n_num = n_num, indexes = indexes));
}

# Equivalent option
count_numeric <- function(...){
  arguments <- list(...);
  indexes <- as.numeric(which(sapply(arguments, class) == "numeric"));
  n_num <- length(indexes);
  return(list(n_num = n_num, indexes = indexes));
}


#### (**) Exercise 7: Try to replicate the plot given in quiz2_plot_2.png together  #### 
#### with this quiz using ggplot functions. Bear in mind that the values plotted come 
#### from the mtcars dataset.

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



#### (**) Exercise 8: Open the R Markdown 'quiz2_rmarkdown.Rmd' given with this file.  ####
#### Make the following modifications to this R Markdown

#### 1) Modify the header to set the output to be a word_document.
#### 2) Modify the general configuration chunk so any R code is not evaluated
#### by default.
#### 3) Create a new text block at the bottom with a header of a smaller size than the one
#### already present in the R Markdown.
#### 4) Create a new code block below the previous text block to compute
#### the maximum value of the mpg column of the mtcars dataset. Bear in mind that this
#### value is not going to be actually computed due to the evaluation set to false in the
#### general configuration. This behavior is expected and correct but the CODE MUST BE PRINTED
#### in the output document.


#### You have to submit the modified 'quiz2_rmarkdown.Rmd' together with this .R file.


#### (***) Exercise 9: Modify the function in auxiliary_functions.R so now it returns ####
####  the 0.75-percentile or third quartile of the argument x instead of the 
#### 0.5-percentile or median. Then use a dt[i, j, by] operation to call this
#### function to compute the third quartile of all the columns in mtcars and store 
#### the output in a variable called 'third_quartiles'. Use of .SD will imply a higher score.
#### The output should look like this

#   mpg    cyl   disp     hp    drat     wt   qsec     vs     am   gear   carb 
#  22.80   8.00 326.00 180.00   3.92   3.61  18.90   1.00   1.00   4.00   4.00 


my_percentile <- function(x){
  quantile(x, 0.75);
}

dt <- as.data.table(mtcars);

third_quartiles <- dt[, sapply(.SD, my_percentile)];
names(third_quartiles) <- colnames(mtcars);
third_quartiles
#### (***) Exercise 10: Open the Shiny app in folder 'quiz2_shiny'.  ####
#### Make the following modifications to the ui.R and server.R files

#### 1) Create a new selectInput called 'group' where an user can
#### choose one or more columns of mtcars.
#### 2) Use the value of this 'group' input to print a DT table of the max values of mpg 
#### in mtcars grouped by the 'group' columns. Suggestion: You should probably use a command 
#### like dt[i, j, by = eval()] to make this work. Notice the eval() function; it is relevant.
#### 3) If a user changes the values of 'group' selected the DT table
#### should also automatically change.
#### 4) You can print this DT table at any position in your shiny app.


#### You have to submit this 'quiz2_shiny' folder together with this .R file.

#########################################################################################################################

#### (*) Exercise 1: Create a function, called 'my_grep', that returns as output  ####
#### TRUE only if the first argument, named 'x', contains the pattern indicated by
#### the second argument, named 'y'. Use default values so that the output of

my_grep(x = "aaaaaa");


#### is TRUE

my_grep <- function(x, y = "a") {
  result <- grepl(y, x);
  return(result);
}



#### (*) Exercise 2: You have the following vector ####

v <- sample(1:100, 100, replace = TRUE);


#### Create a loop that iterates through all the positions of 'v' and, for each ####
#### iteration, prints the value of 'v' in that position if it is a multiple of 5.
#### Do this using both a for and a while loop. Each loop should contain an if
#### (or if-else) clause.

for (index in 1:length(v)){
  if ((v[index] %% 5) == 0){
    print(v[index]);
  } else {
  }
}

index <- 1;
while (index <= length(v)){
  if ((v[index] %% 5) == 0){
    print(v[index]);
  } else {
  }
  index <- index + 1;
}


#### (*) Exercise 3: For this exercise you will be using the same vector 'v' as in exercise 2. ####
#### Your objective now is to create a loop (you can choose which type) that iterates through the positions
#### of 'v' until it finds an index where both 'v[index]' and 'index' are even numbers, in that moment you should
#### print the value of 'v[index]' and the value of 'index' and stop the loop using the "break" command. 
#### Beware infinite loops.


for (index in 1:length(v)){
  if ((v[index] %% 2) == 0 & (index %% 2 == 0)){
    print(index);
    print(v[index]);
    break;
  } 
}

#### (*) Exercise 4: Source the file "auxiliary_functions.R" that comes ####
#### with this test. Compute the function inside this script for each column 
#### of the following dummy dataset and store the result in a variable. 

df <- data.frame(v1 =  sample(c(letters, NA), 100, replace = TRUE),
                  v2 = sample(c(letters, NA), 100, replace = TRUE),
                  v3 = sample(c(letters, NA), 100, replace = TRUE));


#### Do this using both a while loop and sapply(). To which class belongs the
#### resulting variable in each case? (QUESTION 1)

source('C:/Users/Falendor/Dropbox/Planes/proyectos/academia/IE/sessions/13c/auxiliary_functions.R');

ret_1 <- c();
n_col <- 1;
while (n_col <= ncol(df)){
  ret_1 <- c(ret_1, null_counter(df[, n_col]));
  n_col <- n_col + 1;
}
ret_1;

ret_2 <- sapply(df, null_counter);

class(ret_1);
class(ret_2);

#### ANSWER 1: integer for both methods.

#### (*) Exercise 5: You have the following vector ####

v <- c(sample(0:100, 100, replace = TRUE), 500);

#### Try to replicate the plot given in quiz2_plot.png together with this quiz using
#### basic plot functions (not ggplot). Bear in mind that the actual values plotted are going
#### to be probably different in your plot as 'v' is a random vector.

boxplot(v, col = "yellow", 
     main = "my_blox", sub = "subtitle", xlab = "x", 
     ylab = "value",
     ylim = c(-100,500),
     las = 3, cex.axis = 1,
     outline = FALSE);


#### (**) Exercise 6: Create a function called 'count_trues_falses' which takes advantage of  ####
#### the ... argument functionality to return a list with two variables: n_trues counting the
#### number of TRUE values passed to the function and n_falses, counting the FALSE values.
#### For instance, the following call

count_trues_falses(v1 = TRUE, 
             v2 = letters[1:4], 
             v3 = c(TRUE, FALSE, TRUE),
             v4 = c(FALSE),
             v5 = factor(c("cat", "dog", "dog"), levels = c("cat", "dog")));

#### should give this output

# $`n_trues`
# [1] 3
# 
# $n_falses
# [1] 2

count_trues_falses <- function(...){
  arguments <- list(...);
  n_trues <- 0;
  n_falses <- 0;
  for (var in arguments){
    n_trues <- n_trues + sum(var == TRUE);
    n_falses <- n_falses + sum(var == FALSE);
  }
  return(list(n_trues = n_trues, n_falses = n_falses));
}


#### (**) Exercise 7: Try to replicate the plot given in quiz2_plot_2.png together  #### 
#### with this quiz using ggplot functions. Bear in mind that the values plotted come 
#### from the mtcars dataset.

library(ggplot2);
library(data.table);

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



#### (**) Exercise 8: Open the R Markdown 'quiz2_rmarkdown.Rmd' given with this file.  ####
#### Make the following modifications to this R Markdown

#### 1) Modify the header to include a new parameter called 'input' with a default
#### value of 1.
#### 2) Modify the general configuration chunk so for any R block only
#### the code is shown by default.
#### 3) Create a new text block at the bottom with an URL link. 
#### 4) Create a new code block below the previous text block to compute
#### number of rows of mtcars.


#### You have to submit the modified 'quiz2_rmarkdown.Rmd' together with this .R file.


#### (***) Exercise 9: Modify the function in auxiliary_functions.R so now it returns the ####
#### number of values greater than 5 in x instead of the number of missing values. Then use a 
#### dt[i, j, by] operation to call this function to compute the number of non-missing values  
#### for all the columns in mtcars and store the output in a variable called 'large_values'.
#### Use of .SD will imply a higher score. The output should look like this

# mpg  cyl disp   hp drat   wt qsec   vs   am gear carb 
# 32   21   32   32    0    3   32    0    0    0    2 


null_counter <- function(x) {
  ret <- sum(x > 5);
  return(ret);
}

dt <- as.data.table(mtcars);

large_values <- dt[, sapply(.SD, null_counter)];
names(large_values) <- colnames(mtcars);

#### (***) Exercise 10: Open the Shiny app in folder 'quiz2_shiny'.  ####
#### Make the following modifications to the ui.R and server.R files

#### 1) Create a new multiple selectInput called 'rows' where an user can
#### choose a set of rows to filter mtcars.
#### 2) Use the value of this 'filter' input to print a DT table of mtcars for 
#### the rows selected in 'rows'. Rows should be in their original order, i.e row 3
#### should be shown above row 7, etc. Suggestion: You should probably use a command like
#### dt[i, j, by] to make this work.
#### 3) If a user changes the values of 'rows' selected 
#### the DT table should also automatically change.
#### 4) You can print this DT table at any position in your shiny app.


#### You have to submit this 'quiz2_shiny' folder together with this .R file.


### [Samplequiz2] ##########################################################################################

#### (*) Exercise 1: Together with this quiz you received two files (in folder "files"). Read both ####
#### of them using read.table and fread and store the results in four different variables. Bear in mind that
#### file 1 and file represent the same dataset.

library(data.table);
folder_path <- "/Users/Falendor/Dropbox/Planes/proyectos/academia/IE/sessions/20/sample/files";

file1_a <- fread(file.path(folder_path, "file1.csv"));
file1_a;
file1_b <- read.table(file.path(folder_path, "file1.csv"), sep = ";");
file1_b;

file2_a <- fread(file.path(folder_path, "file2.csv"), select = 2:4);
file2_a;
file2_b <- read.table(file.path(folder_path, "file2.csv"), sep = ",", header = TRUE,
                      colClasses=c("NULL",NA,NA,NA));
file2_b;




#### (*) Exercise 2: You have the following data.frame ####

df <- data.frame(name = c("Arsenal", "Chelsea", "Man.City", "Liverpool"),
                 position = c(5, 4, 2, 1),
                 london_based = c(TRUE, TRUE, FALSE, FALSE));

#### Use any indexation method to get the position of Arsenal

df[1,]$position;

### Use indexation by variable ($ operator) in combination with logical indexation
### to get the name of all the teams based on London.

df[df$london_based,]$name;


#### (*) Exercise 3: Cast the mtcars dataset to a data.table. Then, compute the minimum value ####
#### of 'mpg' for each value of 'am', taking into account only cars with more than 4 cylinders.
#### Use only one command with the format dt[i, j, by] (maybe you do not need to use all
#### of the three dimensions or indexes).

dt <- as.data.table(mtcars);

View(mtcars)

dt[cyl >4, min(.SD[,"mpg"]), by = "am"];


#### (*) Exercise 4: You have the following vector ####

v <- rep(c(TRUE,FALSE),100)
v

#### Create a for loop that iterates through all the values of 'v' printing them until
#### it finds a value greater than 90, when you should print "Found in x", where x is
#### the actual index, and stop. Use break.

for (index in 1:3){
  if (v[index] > 90){
    print(sprintf("Found in %s", index));
    next;
  } else {
    print(v[index]);
  }
}

#### (*) Exercise 5: Try to replicate the plot given in final_quiz_plot.png together  #### 
#### with this quiz using ggplot functions. Bear in mind that the values plotted come 
#### from the mtcars dataset.

library(ggplot2);

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

#### (**) Exercise 6: Create a new R Markdown and make the following ####
#### modifications:  

#### 1) Modify the header to include a new parameter called 'class' with a default
#### value of "setosa".
#### 2) Modify the general configuration chunk so for any R block
#### the code is not shown by default.
#### 3) Create a new text block at the bottom with an URL link. 
#### 4) Create a new code block below the previous text block to compute the
#### number of rows in iris where Species is equal to the parameter 'class'.
#### The code must be shown in the output.


#### You have to submit the modified .Rmd together with this .R file.


#### (**) Exercise 7: Compute k-means on mtcars to split the dataset into ####
#### two different clusters. Use as centroids the cars with maximum and minimum
#### weight. Visualize the clusters using clusplot.

library(cluster);

min_wt <- mtcars[which.min(mtcars$wt),];
max_wt <- mtcars[which.max(mtcars$wt),];

clustering <- kmeans(x = mtcars, centers = rbind(min_wt, max_wt));
clusplot(mtcars, clustering$cluster, color=TRUE, shade=FALSE, 
         labels=0, lines=0,  main = "k-means");


#### (**) Exercise 8: Create a new shiny app and make the following modifications ####
#### to the ui.R and server.R files 

#### 1) Create a new single selectInput called 'dataset' where an user can
#### choose between mtcars or iris dataset.
#### 2) Use the value of this 'dataset' input to print a DT table of the selected for 
#### dataset.
#### 3) If a user changes the value of 'dataset' selected the DT table should
#### also automatically change.
#### 4) You can print this DT table at any position in your shiny app.


#### You have to submit this shiny app together with this .R file.

#### (***) Exercise 9: Check the following block of code

index_selected <- c();
for (col in colnames(mtcars)){
  values <- as.numeric(mtcars[, col]);
  correlation <- abs(cor(values, mtcars$wt));
  if (correlation > 0.75){
    index_selected <- c(index_selected, TRUE);
  } else {
    index_selected <- c(index_selected, FALSE);
  }
}
vars_selected <- colnames(mtcars)[index_selected];
vars_selected;

#### Carry out these steps:

#### 1) Create a function 'select_variables' to compute the same operations for
####  a given vector as each iteration of the previous loop is doing for each 
#### column of mtcars.

select_variables <- function(x){
  correlation <- abs(cor(as.numeric(x), mtcars$wt));
  if (correlation > 0.75){
    ret <- TRUE;
  } else {
    ret <- FALSE;
  }
  return(ret);
}

#### 2) Use this function together with any vectorization method to get 
#### the same output as the one in 'selected'

index_selected <- sapply(mtcars, select_variables);
vars_selected_2 <- colnames(mtcars)[index_selected];
identical(vars_selected, vars_selected_2);

#### 3) Try to generalize your function so this operation can be called
#### with any other combination of target (instead of mtcars$wt)
#### and threshold (instead of 0.75). For instance, this code

index_selected <- sapply(iris, select_variables, iris$Petal.Length, 0.9);
colnames(iris)[index_selected];

#### should give the following output

# [1] "Petal.Length" "Petal.Width"  "Species"    

select_variables <- function(x, target, theshold){
  correlation <- abs(cor(as.numeric(x), target));
  if (correlation > theshold){
    ret <- TRUE;
  } else {
    ret <- FALSE;
  }
  return(ret);
}

#### (***) Exercise 10: Your goal now is to predict the 'mpg' column of mtcars. ####
#### Find a model with less than 2.00 of MAE for the test set.

library(e1071);

# setting seed to reproduce results of random sampling
set.seed(100); 

# Convert mtcars to data.table
dat <- as.data.table(mtcars);

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

# Start cluster
library(foreach);
library(doParallel);
stopImplicitCluster();
registerDoParallel(cores = detectCores());

### Define grid
c_values <- seq(from = 0, to = 100, length.out = 10);
eps_values <- seq(from = 10^-6, to = 10^-4, length.out = 10);
gamma_values <- seq(from = 1.0, to = 1.5, length.out = 10);

### Compute grid search
grid_results <-  foreach (c = c_values, .combine = rbind)%:%
  foreach (eps = eps_values, .combine = rbind)%:%
  foreach (gamma = gamma_values, .combine = rbind)%dopar%{
    library(e1071);
    library(data.table);
    
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
    mae_train <- round(mean(abs(errors_train)), 2);
    mae_val <- round(mean(abs(errors_val)), 2);
    
    # Build comparison table
    data.table(c = c, eps = eps, gamma = gamma, 
               mae_train = mae_train,
               mae_val = mae_val);
  }

# Order results by increasing mse and mae
grid_results <- grid_results[order(mae_val, mae_train)];

# Check results
best <- grid_results[1];

### Train final model
# train SVM model with best found set of hyperparamets
model <- svm(mpg ~ ., data = train, kernel="radial",
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
mae_train <- round(mean(abs(errors_train)), 2);
mae_val <- round(mean(abs(errors_val)), 2);
mae_test <- round(mean(abs(errors_test)), 2);

## Summary
sprintf("MAE_train = %s - MAE_val = %s - MAE_test = %s", mae_train, mae_val, mae_test);
