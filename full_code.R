dat<- read.csv('imports-85_data.csv',header = TRUE)
dat[dat==""] <- NA






# 1. Visualizing & Preprocessing

summary(dat)
str(dat)
dat<-dat[,-8]




# NA 존재하는 column 및 row 확인
##"num.of.doors","bore","stroke" 변수에 NA값 존재 
colnames(dat)[ apply(dat, 2, anyNA) ]
sum(is.na(dat$num.of.doors)) ##1개
sum(is.na(dat$bore))  ##4개
sum(is.na(dat$stroke))  ##4개


which(is.na(dat$num.of.doors))
which(is.na(dat$bore))
which(is.na(dat$stroke))





##NA를 포함하고 있는 데이터

dat[c(18,40,41,42,43),]

##missForest: implementation of random forest algorithm
#install.packages("missForest")
library(missForest)

#impute missing values, using all parameters as default values
set.seed(123)
dat_imp <- missForest(dat)

#check imputed values
#dat_imp$ximp

#check imputation error
dat_imp$OOBerror

dat_imputed<-dat_imp$ximp
dat_imputed[c(18,40,41,42,43),]
dat[c(18,40,41,42,43),]





## 연속형 변수, 범주형 변수 구분하여 컬럼명 저장하는 리스트 생성
column_list=colnames(dat_imputed)
continuous_list=c()
category_list=c()

class(dat_imputed[,c('make')])

for(i in column_list){
  class_var=class(dat_imputed[,i])
  if(class_var=='factor'){
    category_list=c(category_list,i)
  }
  else{continuous_list=c(continuous_list,i)}
}

#install.packages('GGally')
library(GGally)
#ggpairs(dat_imputed[,continuous_list])




# corrplot 이용하여 연속형 변수들 사이의 correlation 확인
library(corrplot)
M<-cor(dat_imputed[,continuous_list])
M
corrplot(M,order="FPC",method="number")
## 1. curb.weight, length, width, price, engine.size / 2. city.mpg, highway.mpg





# PCA 차원 축소


ggpairs(dat_imputed[,c('curb.weight','length','width','price','engine.size')])
## 대체적으로 선형적인 관계성을 가지고 있음. ->PCA 사용
ggpairs(dat_imputed[,c('city.mpg','highway.mpg')])

## https://rpubs.com/Evan_Jung/pca
pca_1 <- prcomp(dat_imputed[,c('curb.weight','length','width','price','engine.size')], scale = TRUE)
print(pca_1)
summary(pca_1)
screeplot(pca_1, main = "", type = "lines", pch = 1, npcs = length(pca_1$sdev)) ##2개까지 사용
pca_1_val=predict(pca_1)
pca_1_val=pca_1_val[,c(1,2)]

pca_2 <- prcomp(dat_imputed[,c('city.mpg','highway.mpg')], scale = TRUE)
print(pca_2)
summary(pca_2) ##1개만 사용
pca_2_val=predict(pca_2)
PCA2_2=pca_2_val[,1]

dat_pca <- subset(dat_imputed, select = -c(curb.weight,length,width,
                                           price,engine.size,city.mpg,highway.mpg))
dat_pca<-cbind(dat_pca,pca_1_val,PCA2_2)




# Random Forest Importance Plot

library(caret)
library(dplyr)
library(randomForest)
set.seed(123)

result <- NULL
for(i in seq(4,20, by = 4)){
  tunegrid = expand.grid(.mtry=i)
  set.seed(123)
  rf<- train(normalized_losses ~ ., data=dat_pca, method="rf", metric="RMSE", 
             tuneGrid=tunegrid, ntree=300)
  result <- c(result, rf$results[2])
}
result_vec <- unlist(result)
result_df <- data.frame(mtry = seq(4,20, by = 4), RMSE = result_vec)
result_df %>% ggplot(aes(x = mtry, y =RMSE)) + geom_line(size = 1.1, col = 'khaki4') + geom_point(size = 2, col = 'khaki4') + theme_minimal()

set.seed(123)
dat_rf <- randomForest(normalized_losses ~ ., data=dat_pca, mtry = 12, ntree = 500, importance = T)
varImpPlot(dat_rf)
##상위권 5개-> make, height, num of doors, drive.wheels,PC1,PC2

#install.packages('MLmetrics')
library(MLmetrics)
pred_rf<-predict(dat_rf,dat_pca[,-1])
MSE(dat_pca[,1],pred_rf) ##48.85586




column_list=colnames(dat_pca)
continuous_list=c()
category_list=c()

for(i in column_list){
  class_var=class(dat_pca[,i])
  if(class_var=='factor'){
    category_list=c(category_list,i)
  }
  else{continuous_list=c(continuous_list,i)}
}

continuous_list=continuous_list[-1]

dat_pca[,continuous_list]=scale(dat_pca[,continuous_list])


cont_dat<-dat_pca[,continuous_list]
cat_dat<-dat_pca[,category_list]





# 변수 클러스터링(hclustvar)

#install.packages('ClustOfVar')
library(ClustOfVar)
cont_dat=cont_dat[,-c(2,8,9)] ##height, PC1,PC2 제외
cat_dat=cat_dat[,-c(1,4,6)] ##make, drive,wheels, num.of.doors
tree <- hclustvar(cont_dat,cat_dat)
plot(tree)
groups <- cutree(tree, k=4) 
rect.hclust(tree, k=4, border="red")

groups
names(groups)
group1<-names(groups[groups==1])
group2<-names(groups[groups==2])
group3<-names(groups[groups==3])
group4<-names(groups[groups==4])

group1_dat<-dat_pca[,group1] #cat+cont
group2_dat<-dat_pca[,group2] #cat+cont
group3_dat<-dat_pca[,group3] #cat+cont
group4_dat<-dat_pca[,group4] #cat only


colnames(dat_pca)
final_dat<-dat_pca[,c('normalized_losses','height','make','drive.wheels','num.of.doors','PC1','PC2')]






# 나머지 범주+연속형 변수를 이용하여 변수 클러스터링 진행

library(cluster)
gower_dist<-daisy(group1_dat, metric="gower",stand=T)
sil_width<-c(NA)
for(i in 2:12){
  pam_fit <- pam(gower_dist,diss = TRUE,k = i)
  sil_width[i] <- pam_fit$silinfo$avg.width
}
library(ggplot2)
df <- data.frame(x = 1:12, y = sil_width)
df %>% ggplot(aes(x = x, y = y)) + geom_point(size = 3, col = 'red') + geom_line( size = 1) 

pam_fit1 <- pam(gower_dist, diss = TRUE, k = 2)
group1_dat[pam_fit1$medoids,]
final_dat$cluster_group1 <- pam_fit1$clustering


gower_dist<-daisy(group2_dat, metric="gower",stand=T)
sil_width<-c(NA)
for(i in 2:20){
  pam_fit <- pam(gower_dist,diss = TRUE,k = i)
  sil_width[i] <- pam_fit$silinfo$avg.width
}
library(ggplot2)
df <- data.frame(x = 1:20, y = sil_width)
df %>% ggplot(aes(x = x, y = y)) + geom_point(size = 3, col = 'red') + geom_line( size = 1) 
sil_width[4]
pam_fit2 <- pam(gower_dist, diss = TRUE, k = 4)
group2_dat[pam_fit2$medoids,]
final_dat$cluster_group2 <- pam_fit2$clustering


gower_dist<-daisy(group3_dat, metric="gower",stand=T)
sil_width<-c(NA)
for(i in 2:20){
  pam_fit <- pam(gower_dist,diss = TRUE,k = i)
  sil_width[i] <- pam_fit$silinfo$avg.width
}
library(ggplot2)
df <- data.frame(x = 1:20, y = sil_width)
df %>% ggplot(aes(x = x, y = y)) + geom_point(size = 3, col = 'red') + geom_line( size = 1) 
sil_width[5]
pam_fit3 <- pam(gower_dist, diss = TRUE, k = 5)
group3_dat[pam_fit3$medoids,]
final_dat$cluster_group3 <- pam_fit3$clustering


gower_dist<-daisy(group4_dat, metric="gower",stand=T)
sil_width<-c(NA)
for(i in 2:20){
  pam_fit <- pam(gower_dist,diss = TRUE,k = i)
  sil_width[i] <- pam_fit$silinfo$avg.width
}
df <- data.frame(x = 1:20, y = sil_width)
df %>% ggplot(aes(x = x, y = y)) + geom_point(size = 3, col = 'red') + geom_line( size = 1) 
sil_width[5]
pam_fit4 <- pam(gower_dist, diss = TRUE, k = 5)
group4_dat[pam_fit3$medoids,]

final_dat$cluster_group4 <- pam_fit4$clustering



str(final_dat)

final_dat$cluster_group1<-as.factor(final_dat$cluster_group1)
final_dat$cluster_group2<-as.factor(final_dat$cluster_group2)
final_dat$cluster_group3<-as.factor(final_dat$cluster_group3)
final_dat$cluster_group4<-as.factor(final_dat$cluster_group4)

### 최종독립변수: height, make, drive.wheels, num.of.doors, PC1, PC2, 
### cluster_group1, cluster_group2, cluster_group3, cluster_group4










#### 2. Modeling & Evaluating


# Multiple Linear Regression
str(final_dat)
ggpairs(final_dat[,c('normalized_losses','height','PC1','PC2')]) ## 선형성이 크게 보이지 않음.

model  <- lm(normalized_losses~., data = final_dat)
summary(model)
str(model)
MSE(final_dat[,1],model$fitted.values) ##243.6395

model  <- lm(normalized_losses~height+make+drive.wheels+num.of.doors+cluster_group4, data = final_dat)
summary(model)
MSE(final_dat[,1],model$fitted.values) ##259.8565
####결과물이 좋지 않음




# Random Forest Algorithm

result <- NULL
for(i in seq(2,10, by = 1)){
  tunegrid = expand.grid(.mtry=i)
  set.seed(123)
  rf<- train(normalized_losses ~ ., data=final_dat, method="rf", metric="RMSE", 
             tuneGrid=tunegrid, ntree=300)
  result <- c(result, rf$results[2])
}
result_vec <- unlist(result)
result_df <- data.frame(mtry = seq(2,10, by = 1), RMSE = result_vec)
result_df 
result_df %>% ggplot(aes(x = mtry, y =RMSE)) + geom_line(size = 1.1, col = 'khaki4') + geom_point(size = 2, col = 'khaki4') + theme_minimal()
##mtry=10일 때 가장 rmse 낮음

set.seed(123)
dat_rf <- randomForest(normalized_losses ~ ., data=final_dat, mtry = 10, ntree = 500, importance = T)
varImpPlot(dat_rf)
pred_rf<-predict(dat_rf,final_dat[,-1])
MSE(final_dat[,1],pred_rf) ##46.32688




# SVM Regression 모델

library(e1071)
set.seed(123)
obj <- tune(svm,normalized_losses~., data = final_dat, 
            ranges = list(gamma = 2^(-5:5), cost = 2^(-5:5)))
print(obj)

set.seed(123)
svm_model_after_tune <- svm(normalized_losses~., data = final_dat, kernel="radial", cost=32, gamma=0.625)
pred <- predict(svm_model_after_tune,newdata=final_dat)
MSE(final_dat[,1],pred) ##10.89811

boxplot(final_dat$normalized_losses)
summary(final_dat$normalized_losses)

