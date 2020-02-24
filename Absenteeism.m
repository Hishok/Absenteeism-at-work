close all;
clc;
clear all;

%% Read in table
%Both categorical and Numerical tables include the response variable
%AbsenteeismTimeInHours

data = readtable('Absenteeism_at_work.xls');
%{
The original dataset had converted the categorical variables into dummy
variables, so splitting the data into Categorical and Numerical variables
were not required.

dataCategorical = readtable('Absenteeism_at_work_categorical.xls');
dataNumerical = readtable('Absenteeism_at_work_numerical.xls');
%}
%% Pre processing of data

% The month column has month = 0. This is an error and we have removed the
% 3 rows. 

 data(~data.MonthOfAbsence,:) = [];

 %Remove the ID column. The ID column contains the employee ID. 
 datanew = removevars(data,{'ID'});

%% Descriptive statistics of Categorical Variables
% This was completed in python and the output can be found in the poster.

%% Descriptive statistics of Numerical Variables
% The numerical variables are: TransportationExpense,
% DistanceFromResidenceToWork, ServiceTime, Age, WorkLoadAverage_Day,
% HitTarget, Pet, Son, Weight, Height, BMI. Need to find Mean, Std Dev, Max
% Min

% TransportationExpense
TransportMean = mean(datanew.TransportationExpense);
TransportStd = std(datanew.TransportationExpense);
TransportMax = max(datanew.TransportationExpense);
TransportMin = min(datanew.TransportationExpense);
TransportSkew = skewness(datanew.TransportationExpense);

%DistanceFromResidenceToWork
DistanceMean  = mean(datanew.DistanceFromResidenceToWork);
DistanceStd = std(datanew.DistanceFromResidenceToWork);
DistanceMax = max(datanew.DistanceFromResidenceToWork);
DistanceMin = min(datanew.DistanceFromResidenceToWork);
DistanceSkew = skewness(datanew.DistanceFromResidenceToWork);

%ServiceTime 
ServiceMean = mean(datanew.ServiceTime);
ServiceStd = std(datanew.ServiceTime);
ServiceMax = max(datanew.ServiceTime);
ServiceMin  = min(datanew.ServiceTime);
ServiceSkew  = skewness(datanew.ServiceTime);

%Age - round to nearest integer
AgeMean = round(mean(datanew.Age));
AgeStd = round(std(datanew.Age));
AgeMax = round(max(datanew.Age));
AgeMin = round(min(datanew.Age));
AgeSkew = skewness(datanew.Age);

%WorkloadAverage_Day
WorkloadMean = mean(datanew.WorkLoadAverage_day);
WorkloadStd = std(datanew.WorkLoadAverage_day);
WorkloadMax = max(datanew.WorkLoadAverage_day);
WorkloadMin = min(datanew.WorkLoadAverage_day);
WorkloadSkew = skewness(datanew.WorkLoadAverage_day);

%HitTarget 
HitMean = mean(datanew.HitTarget);
HitStd = std(datanew.HitTarget);
HitMax = max(datanew.HitTarget);
HitMin = min(datanew.HitTarget);
HitSkew = skewness(datanew.HitTarget);

%Pet - round to nearest integer
PetMean = round(mean(datanew.Pet));
PetStd = round(std(datanew.Pet));
PetMax = round(max(datanew.Pet));
PetMin = round(min(datanew.Pet));
PetSkew = skewness(datanew.Pet);

%Son - round to nearest integer
SonMean = round(mean(datanew.Son));
SonStd = round(std(datanew.Son));
SonMax = round(max(datanew.Son));
SonMin = round(min(datanew.Son));
SonSkew = skewness(datanew.Son);

%Weight 
WeightMean = mean(datanew.Weight);
WeightStd = std(datanew.Weight);
WeightMax = max(datanew.Weight);
WeightMin = min(datanew.Weight);
WeightSkew = skewness(datanew.Weight);

%Height
HeightMean = mean(datanew.Height);
HeightStd = std(datanew.Height);
HeightMax = max(datanew.Height);
HeightMin = min(datanew.Height);
HeightSkew = skewness(datanew.Height);

%BMI
BMIMean = mean(datanew.BodyMassIndex);
BMIStd = std(datanew.BodyMassIndex);
BMIMax = max(datanew.BodyMassIndex);
BMIMin = min(datanew.BodyMassIndex);
BMISkew = skewness(datanew.BodyMassIndex);

%AbsenteeismTimeInHours
AbsentMean = mean(datanew.AbsenteeismTimeInHours);
AbsentStd = std(datanew.AbsenteeismTimeInHours);
AbsentMax = max(datanew.AbsenteeismTimeInHours);
AbsentMin = min(datanew.AbsenteeismTimeInHours);
AbsentSkew = skewness(datanew.AbsenteeismTimeInHours);

%% Correlation plot
% Pearson correlation plot was completed, however we have used a
% correlation heatmap from Python which provides a clearer output.
figure('units','normalized','outerposition',[0 0 1 1])
corrplot(datanew)
title ('Pearson Correlation Plot')

%% Correlation
%This was completed in Python to provide a clearer correlation heatmap
cm = corr(datanew{:,:});
figure, imagesc(cm),colorbar
title('Correlation Heat Map')

%% Collinearity
%Default tolerance for condition index = 30 and variance decomposition =
%0.5
% Collinearity shows if there is a high correlation between independent
% variables. This was done because the correlation plot did not show a high
% correlation between the predictor variables and response variable. Output
% can found in the poster. 

figure('units','normalized','outerposition',[0 0 1 1]);
collintest(datanew,'plot', 'on')

%% Removing highly correlated variables
% By looking at collinearity Hit Target, Weight, Height and BMI have a high
% collinearity. We will not be choosing these variables as these are highly
% correlated.

datanew_remove = removevars(datanew,{'Age','HitTarget','Weight','Height','BodyMassIndex'});

%% Split Data
% A 70:30 split was adopted for the train test split. This was done
% manually.
rng('default') %for reproducibility
[m,n] = size(datanew_remove) ;
P = 0.70 ;
idx = randperm(m)  ;
dataTrain = datanew_remove(idx(1:round(P*m)),:) ; 
dataTest = datanew_remove(idx(round(P*m)+1:end),:) ;

%create train and test model for RF predictions
dataTrainRF = dataTrain;
dataTestRF = dataTest;

%% MODEL 1 - Fit Multiple Linear Regression
% The aim of this is to compare a quadratic fit to a linear fit. 
% Time taken to train the quadratic model as this shows the better
% statistics.
% Warning message can be seen due to data badly scaled. Even when applying
% normalisation the error message can be seen
tic
Mdl_quad = fitglm(dataTrain,'quadratic','Distribution','poisson')
toc
Mdl_lin = fitglm(dataTrain,'linear','Distribution','poisson')

figure
plotResiduals(Mdl_quad,'probability')
title('Normal Probability Plot of Residual for Quadratic GLM')
 
figure
plotResiduals(Mdl_lin,'probability')
title('Normal Probability Plot of Residual for Linear GLM')
 

%% Statistics
%To compute Ordinary and Adjusted R squared

quad_Ord_Rsquared = Mdl_quad.Rsquared.Ordinary
quad_Adj_Rsquared = Mdl_quad.Rsquared.Adjusted
 
lin_Ord_Rsquared = Mdl_lin.Rsquared.Ordinary
lin_Adj_Rsquared = Mdl_lin.Rsquared.Adjusted
%{
Manual calculation for RMSE for both quadratic and linear fit by using SSE
and DFE.
We have commented this out as this is not within the remit of the poster.

%Sum of Squared Errors
Mdl_lin.SSE

%Degrees of Freedom Errors
Mdl_lin.DFE

%Calculate MSE and RMSE using SSE and DFE
MSE_lin_calc = Mdl_lin.SSE/Mdl_lin.DFE
RMSE_lin_calc = sqrt(MSE_lin_calc)

MSE_quad_calc = Mdl_quad.SSE/Mdl_quad.DFE
RMSE_quad_calc = sqrt(MSE_quad_calc)
%}

%% Formula for MLR
% Code below outputs the formula for both Linear and Quadratic fit. The
% output was too long for the poster.
Mdl_lin.Formula

Mdl_quad.Formula

%% Optimisation of MLR using Cross Validation

%Creating variables for cross-validation for GLM and for inputs in Random
%Forest model. This will also be used for prediction purposes. 
X = table2array(dataTrain(:,1:end -1)); %exclude the response variable
y = dataTrain.AbsenteeismTimeInHours;

%Cross-Validation of Quadratic fit
%The output computes the RMSE and MSE for kfold values from 2 to 10.
quadratic = @(Xtr,Ytr, Xte) predict(fitglm(Xtr,Ytr,'quadratic','Distribution','poisson'),Xte);
 
mse_quadratic = zeros(1,9);
rmse_quadratic = zeros(1,9);
 
for i=2:10
    mse_quadratic(i-1) = crossval('mse',X,y,'predfun',quadratic,'kfold',i);
    rmse_quadratic(i-1) = sqrt(mse_quadratic(i-1));
end
 
%Cross-Validation of Linear fit
%The output computes the RMSE and MSE for kfold values from 2 to 10.

linear = @(Xtr,Ytr, Xte) predict(fitglm(Xtr,Ytr,'linear','Distribution','poisson'),Xte);
 
mse_linear = zeros(1,9);
rmse_linear = zeros(1,9);
 
for i = 2:10
    mse_linear(i-1) = crossval('mse',X,y,'predfun',linear,'kfold',i);
    rmse_linear(i-1) = sqrt(mse_linear(i-1));
end

%% Plot
%Plotting RMSE with kfold values to show where the RMSE is at the lowest
%point for both Linear and Quadratic fits. 

figure
plot(rmse_quadratic,'--o', 'LineWidth' ,2) ;
xlabel ('kfold');
ylabel ('RMSE') ;
title('Quadratic GLM - RMSE vs kfold');
 
figure
plot(rmse_linear,'--o', 'LineWidth' ,2) ;
xlabel ('kfold') ;
ylabel ('RMSE') ;
title('Linear GLM - RMSE vs kfold');

%% Predict

pred_lin_train = predict(Mdl_lin,X);
pred_lin_test = predict(Mdl_quad,dataTest);

tic
pred_quad_train = predict(Mdl_quad,X);
pred_quad_test = predict(Mdl_lin,dataTest);
toc

%Combine predicted columns to train and test set 
lin_test_glmpred = horzcat(dataTest,cell2table(num2cell(pred_lin_test)));
lin_train_glmpred = horzcat(dataTrain,cell2table(num2cell(pred_lin_train)));

quad_test_glmpred = horzcat(dataTest,cell2table(num2cell(pred_quad_test)));
quad_train_glmpred = horzcat(dataTrain,cell2table(num2cell(pred_quad_train)));

% Change 'Var1' to 'Predicted hours'
lin_test_glmpred.Properties.VariableNames('Var1') = {'Predicted_Hours'};
lin_train_glmpred.Properties.VariableNames('Var1') = {'Predicted_Hours'};

quad_test_glmpred.Properties.VariableNames('Var1') = {'Predicted_Hours'};
quad_train_glmpred.Properties.VariableNames('Var1') = {'Predicted_Hours'};

%% RMSE
%Calculate RMSE for the new models 
%rmse = sqrt(mean((y - yhat).^2))
%yhat is the predicted and y is the observed data
%The quadratic fit for the MLR model performs better with a lower RMSE
%compared to the linear fit. 

lin_rmse_train = sqrt(mean((lin_train_glmpred.Predicted_Hours-lin_train_glmpred.AbsenteeismTimeInHours).^2))
lin_rmse_test = sqrt(mean((lin_test_glmpred.Predicted_Hours-lin_test_glmpred.AbsenteeismTimeInHours).^2))
 
quad_rmse_train = sqrt(mean((quad_train_glmpred.Predicted_Hours-quad_train_glmpred.AbsenteeismTimeInHours).^2))
quad_rmse_test = sqrt(mean((quad_test_glmpred.Predicted_Hours-quad_test_glmpred.AbsenteeismTimeInHours).^2))

%% Model 2 - Random Forest Regression
% Baseline model creation

% Training a regression ensemble (BASELINE MODEL)
MdlBase = fitrensemble(dataTrainRF,'AbsenteeismTimeInHours');

%Testing the trained regression ensemble
pAbsent = predict(MdlBase,X);

%Baseline model MSE and RMSE

MSEBase = resubLoss(MdlBase)
RMSEBase = sqrt(MSEBase)

%% Optimisation of RF Regression
% After running Grid Search, Random Search, Bayesian Optimisation and Cross
% Validation Optimisation, we decided on Bayesian Opt to optimise our RF
% model. Please refrain from running this code as it takes a long time to
% run.

%% Bayesian Optimisation - please do not run as it takes some time.

%{
rng('default')
RF_Bayes_Opt = fitrensemble(dataTrainRF,'AbsenteeismTimeInHours','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));
VariableDescriptions = hyperparameters('fitrensemble',dataTrainRF,'AbsenteeismTimeInHours','Tree');
rng('default')
RF_Bayes_Opt = fitrensemble(dataTrainRF,'AbsenteeismTimeInHours','Method','Bag','OptimizeHyperparameters',VariableDescriptions,'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));
 %}

%Random Search- Bayesian Optimisation was chosen.
%{
rng('default')
t = templateTree('Reproducible',true);
Mdl = fitrensemble(dataTrain,'AbsenteeismTimeInHours','OptimizeHyperparameters','auto','Learners',t, ...
    'HyperparameterOptimizationOptions',struct('optimizer','randomsearch','AcquisitionFunctionName','expected-improvement-plus'))
%}

%% FINAL RANDOM FOREST MODEL AFTER INCORPORATING OPTIMISED HYPERPARAMETERS
% The hyperparameters LSBoost, MinLeafSize and NumLearningCycles have been chosen
% after Bayesian Optimisation.
% It is not possible to calculate the R squared statistic for Random Forest
% regression, hence a comparison has not been done with the MLR model using
% the R squared statistic. 
tic
learns = templateTree('MinLeafSize',1);
numTrees = 320;
RFMDL = fitrensemble(dataTrainRF,'AbsenteeismTimeInHours','Method','LSBoost','Learners',learns,'NumLearningCycles',numTrees);
toc

rf_crossval = crossval(RFMDL,'kfold',10);
RF_Loss = kfoldLoss(rf_crossval,'mode','individual'); 
RF_RMSE = sqrt(RF_Loss); 

%Recalculate MSE and RMSE using Final Model incorporating optimisation
MSEFinalRF = resubLoss(RFMDL)
RMSEFinalRF = sqrt(MSEFinalRF)

%% Plot
%Plotting RMSE vs Learning Cycle (Number of Trees) for 10-fold cross validation
figure
kflc = kfoldLoss(rf_crossval,'Mode','cumulative');
plot(sqrt(kflc)); %Plotting sqrt because kfoldLoss evaluates MSE and not RMSE
xlabel('Learning Cycle','FontSize' ,12);
ylabel('RMSE','FontSize' ,12);
title('Random Forest - RMSE vs Learning Cycles - optimised');
 
%Plotting RMSE vs kfold to check for model performance stability for a 10
%fold cross validation
figure
plot(RF_RMSE,'--o', 'LineWidth' ,1) ;
xlabel ('kfold','FontSize' ,12) ;
ylabel ('RMSE','FontSize' ,12) ;
title('Random Forest - RMSE vs kfold');

%% Prediction
% Using the final model after hyper parameter optimisation to predict the
% number of hours of absenteeism using the training and test data. This
% output will be used to calculate the RMSE for the training and test data.
% 
XRF = table2array(dataTrainRF(:,1:end -1)); %exclude the response variable
yRF = dataTrainRF.AbsenteeismTimeInHours;

%Prediction on the train and test data
tic
RF_Pred = predict(RFMDL,XRF);
RF_Pred_test = predict(RFMDL,dataTestRF);
toc

%Combine predicted columns to train and test set 
RF_Test_prediction = horzcat(dataTestRF,cell2table(num2cell(RF_Pred_test)));
RF_Train_prediction = horzcat(dataTrainRF,cell2table(num2cell(RF_Pred)));

% Change 'Var1' to 'Predicted hours'
RF_Test_prediction.Properties.VariableNames('Var1') = {'Predicted_Hours'};
RF_Train_prediction.Properties.VariableNames('Var1') = {'Predicted_Hours'};

%% RMSE
%Calculate RMSE for the new models 
%rmse = sqrt(mean((y - yhat).^2))
%yhat is the predicted and y is the observed data

RF_rmse_train = sqrt(mean((RF_Train_prediction.Predicted_Hours-RF_Train_prediction.AbsenteeismTimeInHours).^2))
RF_rmse_test = sqrt(mean((RF_Test_prediction.Predicted_Hours-RF_Test_prediction.AbsenteeismTimeInHours).^2))
 
%% Plot RF 
a = 0:120;
b = a;
figure
scatter(yRF,RF_Pred)
xlabel('Actual Absenteeism Time in Hours')
ylabel('Predicted Absenteeism Time in Hours')
title('Actual vs Predicted Random Forest Regression')
hold on 
line(a,b)

%% Plot Quad MLR

a = 0:120;
b = a;
figure
scatter(y,pred_quad_train)
xlabel('Actual Absenteeism Time in Hours')
ylabel('Predicted Absenteeism Time in Hours')
title('Actual vs Predicted Quadratic Multiple Linear Regression')
hold on 
line(a,b)

%% Plot Linear RF 
a = 0:120;
b = a;
figure
scatter(y,pred_lin_train)
xlabel('Actual Absenteeism Time in Hours')
ylabel('Predicted Absenteeism Time in Hours')
title('Actual vs Predicted Linear Multiple Linear Regression')
hold on 
line(a,b)
%% Plot the cumulative, 10-fold cross-validated, mean-squared error (MSE). 

%Cross validating ensemble of regression trees using 10 fold cross validation

RFMDL = fitrensemble(dataTrainRF,'AbsenteeismTimeInHours','Method','LSBoost','Learners',learns,'NumLearningCycles',numTrees,'CrossVal','on');
% Plot the cumulative, 10-fold cross-validated, mean-squared error (MSE). 

kflc = kfoldLoss(RFMDL,'Mode','cumulative');
figure;
plot(kflc);
ylabel('10-fold cross-validated MSE');
xlabel('Learning cycle');

%Display the estimated generalization error of the ensemble.
estGenError = kflc(end)
