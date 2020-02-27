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
