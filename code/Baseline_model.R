##### Benchmark - Vanila Model (partial credits to doi 10.1016/j.ijforecast.2016.01.001 #####

all_data <- read.csv(file = '../data/df_train.csv', header = TRUE, sep = ",", stringsAsFactors = FALSE)

test_data  <- all_data[6529:6552,]
test_data$meanTemp <- 0
test_data$DoW <- 0

train_data <- all_data[1:6528,]
train_data$meanTemp <- 0
train_data$DoW <- 0


##### Day of the Week with 1=Sunday and 7=Saturdat ####
DoW <- function(t){
  # what day of the week is it? 
  temp = weekdays(as.Date(gsub(" ", "-", paste(train_data$month[t],train_data$day[t],train_data$year[t])),'%m-%d-%Y'))

  if(temp == "Sunday"){
    return(1)
  }else if(temp == "Monday"){
    return(2)
  }else if(temp == "Tuesday"){
    return(3)
  }else if(temp == "Wednesday"){
    return(4)
  }else if(temp == "Thursday"){
    return(5)
  }else if(temp == "Friday"){
    return(6)
  }else if(temp == "Saturday"){
    return(7)      
  }else{
    return(NA) 
  }
}


##### Get the average temperture for the given date and hour and the Day of the Week ####
for(counter in 1:nrow(train_data)){
  train_data$meanTemp[counter] <- mean(c(train_data$w1[counter],train_data$w2[counter],train_data$w3[counter],
                                         train_data$w4[counter],train_data$w5[counter],train_data$w6[counter],
                                         train_data$w7[counter],train_data$w8[counter],train_data$w9[counter],
                                         train_data$w10[counter],train_data$w11[counter],train_data$w12[counter],
                                         train_data$w13[counter],train_data$w14[counter],train_data$w15[counter],
                                         train_data$w16[counter],train_data$w17[counter],train_data$w18[counter],
                                         train_data$w19[counter],train_data$w20[counter],train_data$w21[counter],
                                         train_data$w22[counter],train_data$w23[counter],train_data$w24[counter],
                                         train_data$w25[counter]))
  train_data$DoW[counter] <- DoW(counter)
}

for(counter in 1:nrow(test_data)){
  test_data$meanTemp[counter] <- mean(c(test_data$w1[counter],test_data$w2[counter],test_data$w3[counter],
                                        test_data$w4[counter],test_data$w5[counter],test_data$w6[counter],
                                         test_data$w7[counter],test_data$w8[counter],test_data$w9[counter],
                                         test_data$w10[counter],test_data$w11[counter],test_data$w12[counter],
                                         test_data$w13[counter],test_data$w14[counter],test_data$w15[counter],
                                         test_data$w16[counter],test_data$w17[counter],test_data$w18[counter],
                                         test_data$w19[counter],test_data$w20[counter],test_data$w21[counter],
                                         test_data$w22[counter],test_data$w23[counter],test_data$w24[counter],
                                         test_data$w25[counter]))
  test_data$DoW[counter] <- DoW(counter)
}

##### Model Fitting ####

model <- lm(train_data$LOAD ~train_data$month + train_data$DoW+train_data$hour+train_data$DoW*train_data$hour+
              train_data$meanTemp +I(train_data$meanTemp^2)+I(train_data$meanTemp^3)+train_data$meanTemp*train_data$month + 
              I(train_data$meanTemp^2)*train_data$month+I(train_data$meanTemp^3)*train_data$month+
              train_data$meanTemp*train_data$hour+I(train_data$meanTemp^2)*train_data$hour+I(train_data$meanTemp^3)
             *train_data$hour)


summary(model)

# want to predict 30-sep-2010 (all hours)
prediction <- 0
prediction[1:nrow(test_data)] <- 0

for(counter in 1:nrow(test_data)){
  
  prediction[counter] <- model$coefficients[[1]]+model$coefficients[[2]]*test_data$month[counter]+model$coefficients[[3]]*test_data$DoW[counter]+
    model$coefficients[[4]]*test_data$hour[counter]+model$coefficients[[5]]*test_data$meanTemp[counter]+model$coefficients[[6]]*(test_data$meanTemp[counter]^2)+
    model$coefficients[[7]]*(test_data$meanTemp[counter]^3)+model$coefficients[[8]]*test_data$DoW[counter]*test_data$hour[counter]+
    model$coefficients[[9]]*test_data$meanTemp[counter]*test_data$month[counter]+model$coefficients[[10]]*(test_data$meanTemp[counter]^2)*test_data$month[counter]+
    model$coefficients[[11]]*(test_data$meanTemp[counter]^3)*test_data$month[counter]+model$coefficients[[12]]*test_data$meanTemp[counter]*test_data$hour[counter]+
    model$coefficients[[13]]*(test_data$meanTemp[counter]^2)*test_data$hour[counter]+
    model$coefficients[[14]]*(test_data$meanTemp[counter]^3)*test_data$hour[counter]
  
}

# Plotting 
x=1:nrow(test_data)
plot(x,prediction,type='b',pch=1,ylim=c(100,200),col="blue",xlab="Time (in Hours) for 30 September 2010",ylab="Energy Load",main = "Baseline Model vs Test Data")
lines(test_data$LOAD,type='b',pch=3,col="green")
legend("topright", pch = c(1, 3),  col = c("blue", "green"), legend = c("Baseline Model", "Test Data"))
