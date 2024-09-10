clear; close all; clc;

% =========================================================================
% LOAD AND PREPARE DATA
csv_in = 'C:\Users\mages\OneDrive\Documents\mahesh Dissertation\2024-06 (1).csv';
dum = importdata(csv_in, ',');
series = dum.textdata(1, 2:end);
tcode = dum.data(1, :);
rawdata = dum.data(2:end, :);

final_datevec = datevec(dum.textdata(end, 1));
final_month = final_datevec(2);
final_year = final_datevec(1);

dates = (1959 + 1/12:1/12:final_year + final_month/12)';
T = size(dates, 1);
rawdata = rawdata(1:T, :);

% =========================================================================
% PROCESS DATA
yt = prepare_missing(rawdata, tcode);
yt = yt(3:T, :);
dates = dates(3:T, :);
[data, n] = remove_outliers(yt);

% =========================================================================
% ESTIMATE FACTORS
DEMEAN = 2; jj = 2; kmax = 8;
[ehat, Fhat, lamhat, ve2, x2] = factors_em(data, kmax, jj, DEMEAN);

% =========================================================================
% FEATURE ENGINEERING
X = Fhat;
X_poly = [X, X.^2]; % Adding polynomial features

% =========================================================================
% PREPARE DATA FOR MACHINE LEARNING
ml_data = [dates, X_poly];
y = data(:, 1);

% Handle NaN or infinite values in target variable
y(isnan(y) | isinf(y)) = nanmean(y);

% Split data into training and testing sets (80% train, 20% test)
cv = cvpartition(size(X_poly, 1), 'HoldOut', 0.2);
idx = cv.test;
X_train = X_poly(~idx, :);
y_train = y(~idx);
X_test = X_poly(idx, :);
y_test = y(idx);

% Check for NaNs or infinite values
assert(all(~isnan(X_train(:))) && all(~isinf(X_train(:))), 'X_train contains NaN or Inf values');
assert(all(~isnan(y_train(:))) && all(~isinf(y_train(:))), 'y_train contains NaN or Inf values');

% =========================================================================
% TRAIN MODELS WITH CROSS-VALIDATION

% 1. Random Forest with cross-validation for the number of trees
numTrees = [50, 100, 150, 200];
mse_cv_rf = zeros(length(numTrees), 1);
for i = 1:length(numTrees)
    trees = numTrees(i);
    cvModel = fitrensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', trees, 'KFold', 5);
    mse_cv_rf(i) = kfoldLoss(cvModel);
end
[~, bestIdx_rf] = min(mse_cv_rf);
bestNumTrees_rf = numTrees(bestIdx_rf);
rfModel = fitrensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', bestNumTrees_rf);
y_pred_rf = predict(rfModel, X_test);
mse_rf = mean((y_test - y_pred_rf).^2);
mae_rf = mean(abs(y_test - y_pred_rf)); 
r2_rf = 1 - sum((y_test - y_pred_rf).^2) / sum((y_test - mean(y_test)).^2);

% Calculate feature importance for Random Forest
rf_importance = oobPermutedPredictorImportance(rfModel);

% 2. Gradient Boosting with cross-validation for the number of learning cycles
numCycles = [50, 100, 150, 200];
mse_cv_gb = zeros(length(numCycles), 1);
for i = 1:length(numCycles)
    cycles = numCycles(i);
    cvModel = fitrensemble(X_train, y_train, 'Method', 'LSBoost', 'NumLearningCycles', cycles, 'KFold', 5);
    mse_cv_gb(i) = kfoldLoss(cvModel);
end
[~, bestIdx_gb] = min(mse_cv_gb);
bestNumCycles_gb = numCycles(bestIdx_gb);  % Corrected this line
gbModel = fitrensemble(X_train, y_train, 'Method', 'LSBoost', 'NumLearningCycles', bestNumCycles_gb);
y_pred_gb = predict(gbModel, X_test);
mse_gb = mean((y_test - y_pred_gb).^2);
mae_gb = mean(abs(y_test - y_pred_gb));
r2_gb = 1 - sum((y_test - y_pred_gb).^2) / sum((y_test - mean(y_test)).^2);

% Calculate feature importance for Gradient Boosting
gb_importance = predictorImportance(gbModel);

% 3. Ridge Regression with cross-validation for lambda
lambdas = logspace(-4, 4, 10);
mse_cv_ridge = zeros(length(lambdas), 1);
for i = 1:length(lambdas)
    lambda = lambdas(i);
    cvModel = fitrlinear(X_train, y_train, 'Learner', 'leastsquares', 'Lambda', lambda, 'Regularization', 'ridge', 'KFold', 5);
    mse_cv_ridge(i) = kfoldLoss(cvModel);
end
[~, bestIdx_ridge] = min(mse_cv_ridge);
bestLambda_ridge = lambdas(bestIdx_ridge);
ridgeModel = fitrlinear(X_train, y_train, 'Learner', 'leastsquares', 'Lambda', bestLambda_ridge, 'Regularization', 'ridge');
y_pred_ridge = predict(ridgeModel, X_test);
mse_ridge = mean((y_test - y_pred_ridge).^2);
mae_ridge = mean(abs(y_test - y_pred_ridge));
r2_ridge = 1 - sum((y_test - y_pred_ridge).^2) / sum((y_test - mean(y_test)).^2);

% 4. Lasso Regression with cross-validation for lambda
mse_cv_lasso = zeros(length(lambdas), 1);
for i = 1:length(lambdas)
    lambda = lambdas(i);
    cvModel = fitrlinear(X_train, y_train, 'Learner', 'leastsquares', 'Lambda', lambda, 'Regularization', 'lasso', 'KFold', 5);
    mse_cv_lasso(i) = kfoldLoss(cvModel);
end
[~, bestIdx_lasso] = min(mse_cv_lasso);
bestLambda_lasso = lambdas(bestIdx_lasso);
lassoModel = fitrlinear(X_train, y_train, 'Learner', 'leastsquares', 'Lambda', bestLambda_lasso, 'Regularization', 'lasso');
y_pred_lasso = predict(lassoModel, X_test);
mse_lasso = mean((y_test - y_pred_lasso).^2);
mae_lasso = mean(abs(y_test - y_pred_lasso));
r2_lasso = 1 - sum((y_test - y_pred_lasso).^2) / sum((y_test - mean(y_test)).^2);

% 5. SVM with cross-validation for C parameter
C_values = logspace(-3, 3, 7);
mse_cv_svm = zeros(length(C_values), 1);
for i = 1:length(C_values)
    C = C_values(i);
    cvModel = fitrsvm(X_train, y_train, 'KernelFunction', 'rbf', 'BoxConstraint', C, 'KFold', 5);
    mse_cv_svm(i) = kfoldLoss(cvModel);
end
[~, bestIdx_svm] = min(mse_cv_svm);
bestC_svm = C_values(bestIdx_svm);
svmModel = fitrsvm(X_train, y_train, 'KernelFunction', 'rbf', 'BoxConstraint', bestC_svm, 'Standardize', true);
y_pred_svm = predict(svmModel, X_test);
mse_svm = mean((y_test - y_pred_svm).^2);
mae_svm = mean(abs(y_test - y_pred_svm));
r2_svm = 1 - sum((y_test - y_pred_svm).^2) / sum((y_test - mean(y_test)).^2);

% =========================================================================
% COMPARE MODELS
models = {'Random Forest', 'Gradient Boosting', 'Ridge Regression', 'Lasso Regression', 'SVM'};
mse_values = [mse_rf, mse_gb, mse_ridge, mse_lasso, mse_svm];
mae_values = [mae_rf, mae_gb, mae_ridge, mae_lasso, mae_svm];
r2_values = [r2_rf, r2_gb, r2_ridge, r2_lasso, r2_svm];

% Plot MSE
figure;
bar(mse_values);
set(gca, 'XTickLabel', models);
ylabel('Mean Squared Error');
title('Model Comparison: MSE');

% Plot MAE
figure;
bar(mae_values);
set(gca, 'XTickLabel', models);
ylabel('Mean Absolute Error');
title('Model Comparison: MAE');

% =========================================================================
dates_test = datetime(dates(idx) * 12 * 30, 'ConvertFrom', 'datenum', 'Format', 'yyyy-MM');

% Plot Actual vs Random Forest
figure;
plot(dates_test, y_test, 'b', 'DisplayName', 'Actual Unemployment');
hold on;
plot(dates_test, y_pred_rf, 'r', 'DisplayName', 'Random Forest');
legend('show');
title('Actual vs Random Forest');
xlabel('Date');
ylabel('Unemployment Rate');
hold off;

% Plot Actual vs Gradient Boosting
figure;
plot(dates_test, y_test, 'b', 'DisplayName', 'Actual Unemployment');
hold on;
plot(dates_test, y_pred_gb, 'g', 'DisplayName', 'Gradient Boosting');
legend('show');
title('Actual vs Gradient Boosting');
xlabel('Date');
ylabel('Unemployment Rate');
hold off;

% Plot Actual vs Ridge Regression
figure;
plot(dates_test, y_test, 'b', 'DisplayName', 'Actual Unemployment');
hold on;
plot(dates_test, y_pred_ridge, 'c', 'DisplayName', 'Ridge Regression');
legend('show');
title('Actual vs Ridge Regression');
xlabel('Date');
ylabel('Unemployment Rate');
hold off;

% Plot Actual vs Lasso Regression
figure;
plot(dates_test, y_test, 'b', 'DisplayName', 'Actual Unemployment');
hold on;
plot(dates_test, y_pred_lasso, 'k', 'DisplayName', 'Lasso Regression');
legend('show');
title('Actual vs Lasso Regression');
xlabel('Date');
ylabel('Unemployment Rate');
hold off;

% Plot Actual vs SVM
figure;
plot(dates_test, y_test, 'b', 'DisplayName', 'Actual Unemployment');
hold on;
plot(dates_test, y_pred_svm, 'm', 'DisplayName', 'SVM');
legend('show');
title('Actual vs SVM');
xlabel('Date');
ylabel('Unemployment Rate');
hold off;

% =========================================================================
% Summary Table for MSE, MAE, and R²
T = table(models', mse_values', mae_values', r2_values', ...
    'VariableNames', {'Model', 'MeanSquaredError', 'MeanAbsoluteError', 'R_squared'});

% Display the table
disp(T);

% Optionally, save the table to a file for documentation
writetable(T, 'model_performance_summary_with_svm.csv');

% =========================================================================
% Display Feature Importance for Random Forest and Gradient Boosting
figure;
bar(rf_importance);
set(gca, 'XTickLabel', series(1:size(X_train, 2)));
ylabel('Importance');
title('Feature Importance: Random Forest');

figure;
bar(gb_importance);
set(gca, 'XTickLabel', series(1:size(X_train, 2)));
ylabel('Importance');
title('Feature Importance: Gradient Boosting');
