%% CNN V10 

clc; clear; close all;

if ~exist('MasterData_S1_v7.mat', 'file')
    error('MasterData_S1_v7.mat bulunamadı! Önce Pipeline v5 çalıştır.');
end
load('MasterData_S1_v7.mat');
num_samples = size(X_Mvoid, 4);
fprintf('Toplam veri: %d kesit\n', num_samples);

% =========================================================================
% 1. KİLİTLİ TEST SENARYOLARI
% =========================================================================
hedef_benchmark = [
    5, 300, 40;
    3, 400, 50;
    4, 200, 30
];

is_bench  = ismember(X_Scalars(:,1:3), hedef_benchmark, 'rows');
test_idx  = find(is_bench);
model_idx = find(~is_bench);

fprintf('Kilitli Test : %d kesit\n', length(test_idx));
fprintf('Train+Val    : %d kesit\n', length(model_idx));

% =========================================================================
% 2. NORMALİZASYON
% =========================================================================
Max_S1 = max(Y_S1(:));
Y_Norm = Y_S1 / Max_S1;
fprintf('σ_1 max      : %.2f MPa\n', Max_S1);

% =========================================================================
% 3. 7 KANALLI GİRDİ
% =========================================================================
df_min   = min(X_Scalars(:,5));
df_range = max(X_Scalars(:,5)) - df_min;
if df_range == 0, df_range = 1; end

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
% 4. SENARYO BAZLI TRAIN / VAL AYRIMI
% =========================================================================
kalan_scens = unique(X_Scalars(model_idx, 1:3), 'rows');
n_kalan     = size(kalan_scens, 1);

rng(42);
karisik     = randperm(n_kalan);
n_train     = round(0.85 * n_kalan);
train_scens = kalan_scens(karisik(1:n_train), :);

train_idx = []; val_idx = [];
for i = 1:length(model_idx)
    idx = model_idx(i);
    if ismember(X_Scalars(idx,1:3), train_scens, 'rows')
        train_idx = [train_idx; idx]; %#ok<AGROW>
    else
        val_idx   = [val_idx;   idx]; %#ok<AGROW>
    end
end

XTrain = X_All(:,:,:,train_idx);   YTrain = Y_Norm(:,:,:,train_idx);
XVal   = X_All(:,:,:,val_idx);     YVal   = Y_Norm(:,:,:,val_idx);

fprintf('--> Train : %d kesit (%d senaryo)\n', length(train_idx), n_train);
fprintf('--> Val   : %d kesit (%d senaryo)\n', length(val_idx),   n_kalan-n_train);

% =========================================================================
% 5. MİMARİ — 2 Residual Blok + Sigmoid (geri eklendi)
% =========================================================================
lgraph = layerGraph();

lgraph = addLayers(lgraph, [
    imageInputLayer([5 50 7], 'Name','input', 'Normalization','none')
    convolution2dLayer([3 3], 64, 'Padding','same', 'Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
]);

% Residual Blok 1
lgraph = addLayers(lgraph, [
    convolution2dLayer([3 3], 64, 'Padding','same', 'Name','res1_conv1')
    batchNormalizationLayer('Name','res1_bn1')
    reluLayer('Name','res1_relu1')
    convolution2dLayer([3 3], 64, 'Padding','same', 'Name','res1_conv2')
    batchNormalizationLayer('Name','res1_bn2')
]);
lgraph = addLayers(lgraph, [
    additionLayer(2, 'Name','add1')
    reluLayer('Name','relu_add1')
]);
lgraph = connectLayers(lgraph, 'relu1',    'res1_conv1');
lgraph = connectLayers(lgraph, 'relu1',    'add1/in2');
lgraph = connectLayers(lgraph, 'res1_bn2', 'add1/in1');

% Residual Blok 2
lgraph = addLayers(lgraph, [
    convolution2dLayer([3 3], 64, 'Padding','same', 'Name','res2_conv1')
    batchNormalizationLayer('Name','res2_bn1')
    reluLayer('Name','res2_relu1')
    convolution2dLayer([3 3], 64, 'Padding','same', 'Name','res2_conv2')
    batchNormalizationLayer('Name','res2_bn2')
]);
lgraph = addLayers(lgraph, [
    additionLayer(2, 'Name','add2')
    reluLayer('Name','relu_add2')
]);
lgraph = connectLayers(lgraph, 'relu_add1', 'res2_conv1');
lgraph = connectLayers(lgraph, 'relu_add1', 'add2/in2');
lgraph = connectLayers(lgraph, 'res2_bn2',  'add2/in1');

% Çıkış — sigmoid geri eklendi
lgraph = addLayers(lgraph, [
    convolution2dLayer([1 1], 32, 'Name','final_dense')
    reluLayer('Name','final_relu')
    convolution2dLayer([1 1], 1,  'Name','output_conv')
    sigmoidLayer('Name','sigmoid_out')
    regressionLayer('Name','output')
]);
lgraph = connectLayers(lgraph, 'relu_add2', 'final_dense');

analyzeNetwork(lgraph);

% =========================================================================
% 6. EĞİTİM
% =========================================================================
options = trainingOptions('adam', ...
    'MaxEpochs',           300,                    ...
    'MiniBatchSize',       64,                     ...
    'InitialLearnRate',    1e-3,                   ...
    'LearnRateSchedule',   'piecewise',            ...
    'LearnRateDropFactor', 0.5,                    ...
    'LearnRateDropPeriod', 50,                     ...
    'ValidationData',      {XVal, YVal},           ...
    'ValidationFrequency', 30,                     ...
    'OutputNetwork',       'best-validation-loss', ...
    'Plots',               'training-progress',   ...
    'Verbose',             false,                  ...
    'ExecutionEnvironment','auto');

fprintf('\nEğitim başlıyor...\n');
[net, info] = trainNetwork(XTrain, YTrain, lgraph, options);

save('Trained_CNN_V10.mat', ...
    'net', 'info', 'Max_S1', 'test_idx', 'df_min', 'df_range', '-v7.3');
disp('Kaydedildi: Trained_CNN_V10.mat');

% =========================================================================
% 7. TEST DEĞERLENDİRME
% =========================================================================
disp('Test seti değerlendiriliyor...');

XTest = X_All(:,:,:,test_idx);
YTest = Y_Norm(:,:,:,test_idx);

Y_pred_norm = predict(net, XTest);

% Maske uygula — maske dışı tahminleri sıfırla
mask_test           = XTest(:,:,1,:) > 0.5;
Y_pred_norm_masked  = Y_pred_norm .* mask_test;

% Denormalize → MPa
Y_pred_mpa = Y_pred_norm_masked * Max_S1;
Y_true_mpa = YTest              * Max_S1;

% Sadece maske içi noktalar üzerinden metrik
y_t = double(Y_true_mpa(mask_test));
y_p = double(Y_pred_mpa(mask_test));

rmse = sqrt(mean((y_t - y_p).^2));
mae  = mean(abs(y_t - y_p));
r2   = 1 - sum((y_t-y_p).^2) / sum((y_t-mean(y_t)).^2);

fprintf('\n=== TEST METRİKLERİ — σ_1 ===\n');
fprintf('RMSE : %.4f MPa\n', rmse);
fprintf('MAE  : %.4f MPa\n', mae);
fprintf('R²   : %.6f\n',     r2);

% =========================================================================
% 8. FDM vs CNN CONTOUR (her benchmark senaryo, Stage=4)
% =========================================================================
for s = 1:size(hedef_benchmark, 1)
    H = hedef_benchmark(s,1);
    D = hedef_benchmark(s,2);
    W = hedef_benchmark(s,3);

    scen_mask = (X_Scalars(test_idx,1)==H & ...
                 X_Scalars(test_idx,2)==D & ...
                 X_Scalars(test_idx,3)==W & ...
                 X_Scalars(test_idx,4)==4);
    idx_local = find(scen_mask, 1, 'first');
    if isempty(idx_local), continue; end

    fdm = squeeze(Y_true_mpa(:,:,1,idx_local));
    cnn = squeeze(Y_pred_mpa(:,:,1,idx_local));

    % Maske dışını NaN → görsel daha temiz
    mk       = squeeze(mask_test(:,:,1,idx_local));
    fdm(~mk) = NaN;
    cnn(~mk) = NaN;

    clim = [0, max([fdm(:); cnn(:)], [], 'omitnan')];

    figure('Name', sprintf('σ_1: Tc=%dm D=%dm Wp=%dm', H, D, W), ...
           'NumberTitle','off', 'Position',[100 100 900 380]);

    subplot(2,1,1)
    imagesc(fdm, clim); colormap(jet); colorbar;
    title(sprintf('FDM — σ_1 (MPa) | Tc=%dm D=%dm Wp=%dm Aşama=4', H,D,W), ...
          'FontSize',11);
    xlabel('X (zone)'); ylabel('Z (zone)'); axis equal tight;

    subplot(2,1,2)
    imagesc(cnn, clim); colormap(jet); colorbar;
    title('CNN V9 — σ_1 (MPa)', 'FontSize',11);
    xlabel('X (zone)'); ylabel('Z (zone)'); axis equal tight;
end