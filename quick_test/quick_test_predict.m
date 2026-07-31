%% QUICK TEST — Reproduce reported σ1 surrogate metrics from the trained network
%
% This script loads the trained CNN surrogate (Trained_CNN_V10.mat) and the
% processed dataset (MasterData_S1_v7.mat) included in this folder, rebuilds
% the held-out benchmark test set, runs inference, and reports RMSE / MAE / R²
% for the major principal stress (sigma_1) field.
%
% It performs INFERENCE ONLY (no training) and completes in under a minute
% on CPU. It reproduces the same held-out benchmark scenarios and the same
% evaluation procedure used to obtain the metrics reported in the manuscript.
%
% Requirements: MATLAB with Deep Learning Toolbox.
%
% Usage:
%   >> cd quick_test
%   >> quick_test_predict

clc; clear; close all;

fprintf('Quick test — loading trained network and dataset...\n');
load('Trained_CNN_V10.mat', 'net', 'Max_S1', 'df_min', 'df_range');
load('MasterData_S1_v7.mat', 'X_Mvoid', 'Y_S1', 'X_Scalars');

num_samples = size(X_Mvoid, 4);
fprintf('Loaded %d cross-sections.\n', num_samples);

% =========================================================================
% 1. LOCKED BENCHMARK SCENARIOS (held out from training)
% =========================================================================
hedef_benchmark = [
    5, 300, 40;
    3, 400, 50;
    4, 200, 30
];

is_bench = ismember(X_Scalars(:, 1:3), hedef_benchmark, 'rows');
test_idx = find(is_bench);
fprintf('Benchmark test set: %d cross-sections.\n', length(test_idx));

% =========================================================================
% 2. NORMALIZATION (using the statistics saved at training time)
% =========================================================================
Y_Norm = Y_S1 / Max_S1;

% =========================================================================
% 3. 7-CHANNEL INPUT RECONSTRUCTION (identical to model/CNN_v10.m, Section 3)
% =========================================================================
X_All = zeros(5, 50, 7, num_samples, 'single');
for i = 1:num_samples
    sk = X_Scalars(i, :);
    X_All(:,:,1,i) = X_Mvoid(:,:,1,i);
    X_All(:,:,2,i) = (sk(1) - 3)   / 2;
    X_All(:,:,3,i) = (sk(2) - 200) / 200;
    X_All(:,:,4,i) = (sk(4) - 1)   / 7;
    X_All(:,:,5,i) = (sk(5) - df_min) / df_range;
    X_All(:,:,6,i) = (sk(6) - 1)   / 2;
    X_All(:,:,7,i) = (sk(3) - 30)  / 20;
end

% =========================================================================
% 4. INFERENCE ON THE BENCHMARK TEST SET
% =========================================================================
XTest = X_All(:,:,:,test_idx);
YTest = Y_Norm(:,:,:,test_idx);

fprintf('Running inference...\n');
Y_pred_norm = predict(net, XTest);

mask_test          = XTest(:,:,1,:) > 0.5;
Y_pred_norm_masked = Y_pred_norm .* mask_test;

Y_pred_mpa = Y_pred_norm_masked * Max_S1;
Y_true_mpa = YTest              * Max_S1;

y_t = double(Y_true_mpa(mask_test));
y_p = double(Y_pred_mpa(mask_test));

rmse = sqrt(mean((y_t - y_p).^2));
mae  = mean(abs(y_t - y_p));
r2   = 1 - sum((y_t - y_p).^2) / sum((y_t - mean(y_t)).^2);

fprintf('\n=== QUICK TEST RESULTS — sigma_1 (benchmark test set) ===\n');
fprintf('RMSE : %.4f MPa\n', rmse);
fprintf('MAE  : %.4f MPa\n', mae);
fprintf('R^2  : %.6f\n', r2);
fprintf('\nExpected (manuscript): R^2 ~ 0.95, RMSE ~ 4.70 MPa, MAE ~ 2.83 MPa.\n');
fprintf('Small deviations from run to run are expected due to floating-point\n');
fprintf('and hardware differences; values should match closely.\n');
