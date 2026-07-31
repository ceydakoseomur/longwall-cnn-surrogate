%% DERİN ÖĞRENME DATA PIPELINE v7

clc; clear; close all;

ana_klasor = pwd;
klasorler  = dir('*m_*m_*m_MP3');
klasorler  = klasorler([klasorler.isdir]);

fprintf('Data Pipeline v7 — Kusursuz Katı Maske\n\n');

MAX_ORNEK = 4000;
X_Mvoid   = zeros(5, 50, 1, MAX_ORNEK, 'single');
Y_S1      = zeros(5, 50, 1, MAX_ORNEK, 'single');
X_Scalars = zeros(MAX_ORNEK, 6, 'single');

ornek_sayaci = 0;

for i = 1:length(klasorler)

    klasor_adi = klasorler(i).name;
    degerler   = regexp(klasor_adi, '(\d+)m_(\d+)m_(\d+)m_MP3', 'tokens');
    if isempty(degerler), continue; end

    thickness = str2double(degerler{1}{1});
    depth     = str2double(degerler{1}{2});
    pillar    = str2double(degerler{1}{3});

    fprintf('[%d/%d] İşleniyor: %s\n', i, length(klasorler), klasor_adi);
    hedef_yol = fullfile(ana_klasor, klasor_adi);

    xz_dosyalari = dir(fullfile(hedef_yol, 'out_panel*_part*_rp*_y*.csv'));

    for j = 1:length(xz_dosyalari)

        if ornek_sayaci >= MAX_ORNEK
            warning('MAX_ORNEK doldu (%d).', MAX_ORNEK);
            break;
        end

        dosya_adi  = xz_dosyalari(j).name;
        dosya_yolu = fullfile(hedef_yol, dosya_adi);

        tokens = regexp(dosya_adi, ...
            'out_panel(\d+)_part(\d+)_rp(\d+)_y(\d+)', 'tokens');
        if isempty(tokens), continue; end

        panel    = str2double(tokens{1}{1});
        ham_part = str2double(tokens{1}{2});
        rp       = str2double(tokens{1}{3});
        y_val    = str2double(tokens{1}{4});

        stage    = ham_part + (panel - 1) * 4;
        Panel_Y0 = 300;
        if stage <= 4
            Face_Y = Panel_Y0 + stage * 250;
        else
            Face_Y = Panel_Y0 + (stage - 4) * 250;
        end
        dist_face = y_val - Face_Y;

        xmin = 0;
        if pillar == 30
            if rp==1, xmin=295; elseif rp==2, xmin=585; elseif rp==3, xmin=875; end
        elseif pillar == 40
            if rp==1, xmin=280; elseif rp==2, xmin=580; elseif rp==3, xmin=880; end
        elseif pillar == 50
            if rp==1, xmin=265; elseif rp==2, xmin=575; elseif rp==3, xmin=885; end
        end

        dosya_metni = fileread(dosya_yolu);
        dosya_metni = strrep(dosya_metni, char(13), '');
        bloklar     = strsplit(dosya_metni, 'list zone extra ');

        ids  = []; val50 = []; val5 = [];

        for b = 2:length(bloklar)
            blok    = bloklar{b};
            blok_no = sscanf(blok, '%d', 1);

            [ids_temp, val_temp] = parseBlockRegex(blok);
            if isempty(ids_temp), continue; end

            switch blok_no
                case 50, ids = ids_temp; val50 = val_temp;
                case 5,  val5 = val_temp;
            end
        end

        min_len = min([length(ids), length(val50), length(val5)]);
        if min_len == 0, continue; end

        nx   = pillar;
        zmax = -(depth + 20);
        zmin = zmax - thickness;

        % 1. Grid başlangıcı
        Grid_Mvoid = zeros(5, 50, 1, 'single');
        Grid_S1    = NaN(5, 50, 1, 'single');   % NaN: veri yok

        Z_tavan = zmax - 0.5;
        offset  = floor((50 - nx) / 2);         % tam sayı garantili

        % 2. Kusursuz katı maske — geometrik olarak kesin
        sutun_baslangic = 1 + offset;
        sutun_bitis     = nx + offset;
        Grid_Mvoid(1:5, sutun_baslangic:sutun_bitis, 1) = 1;

        % 3. FLAC3D verilerini yerleştir
        gecerli = 0;
        for k = 1:min_len
            b_val = val50(k);
            if b_val <= 0, continue; end

            k_idx = ceil(b_val / nx);
            j_idx = b_val - (k_idx - 1) * nx;

            Z_k   = zmin + (k_idx - 0.5);
            satir = round(Z_tavan - Z_k) + 1;
            sutun = j_idx + offset;

            if satir >= 1 && satir <= 5 && sutun >= 1 && sutun <= 50
                Grid_S1(satir, sutun, 1) = single(abs(val5(k)) / 1e6);
                gecerli = gecerli + 1;
            end
        end

        if gecerli == 0, continue; end

        % 4. Kontrollü delik onarımı
        tmp_S1 = Grid_S1(:,:,1);
        tmp_S1 = fillmissing(tmp_S1, 'nearest', 2);  % X yönünde
        tmp_S1 = fillmissing(tmp_S1, 'nearest', 1);  % Z yönünde

        % 5. Fiziksel kilit — galeri her zaman sıfır
        tmp_S1(Grid_Mvoid(:,:,1) == 0) = 0;
        tmp_S1(isnan(tmp_S1))          = 0;  % kalan NaN → 0

        Grid_S1(:,:,1) = tmp_S1;

        ornek_sayaci = ornek_sayaci + 1;

        X_Mvoid(:, :, :, ornek_sayaci) = Grid_Mvoid;
        Y_S1(:, :, :, ornek_sayaci)    = Grid_S1;
        X_Scalars(ornek_sayaci, :) = single([thickness, depth, pillar, ...
                                             stage, dist_face, rp]);

    end % dosya döngüsü

    if ornek_sayaci >= MAX_ORNEK, break; end
end % senaryo döngüsü

% =========================================================================
% SON İŞLEMLER + KAYIT
% =========================================================================
X_Mvoid   = X_Mvoid(:, :, :, 1:ornek_sayaci);
Y_S1      = Y_S1(:, :, :, 1:ornek_sayaci);
X_Scalars = X_Scalars(1:ornek_sayaci, :);

fprintf('\n==== İŞLEM TAMAMLANDI ====\n');
fprintf('Toplam Örnek : %d\n', ornek_sayaci);
fprintf('σ_1 aralığı  : [%.2f, %.2f] MPa\n', min(Y_S1(:)), max(Y_S1(:)));

n_outlier = sum(Y_S1(:) > 300);
fprintf('300 MPa üstü : %d\n', n_outlier);
if n_outlier > 0
    warning('Hâlâ aykırı değerler var!');
end

dolu = sum(X_Mvoid(:) > 0);
fprintf('Grid doluluk : %.1f%%\n', 100*dolu/numel(X_Mvoid));

% Offset doğrulama
fprintf('\nOffset Kontrolü:\n');
fprintf('  Wp=30 → offset=%d → kolon %d-%d\n', floor((50-30)/2), 1+floor((50-30)/2), 30+floor((50-30)/2));
fprintf('  Wp=40 → offset=%d → kolon %d-%d\n', floor((50-40)/2), 1+floor((50-40)/2), 40+floor((50-40)/2));
fprintf('  Wp=50 → offset=%d → kolon %d-%d\n', floor((50-50)/2), 1+floor((50-50)/2), 50+floor((50-50)/2));

save('MasterData_S1_v7.mat', 'X_Mvoid', 'Y_S1', 'X_Scalars', '-v7.3');
disp('Kayıt: MasterData_S1_v7.mat');

% =========================================================================
function [ids, values] = parseBlockRegex(blockText)
    lines        = strsplit(blockText, '\n');
    tokens       = regexp(lines, ...
        '^\s*(\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', ...
        'tokens', 'once');
    valid_idx    = ~cellfun('isempty', tokens);
    valid_tokens = tokens(valid_idx);
    if ~isempty(valid_tokens)
        data   = str2double(vertcat(valid_tokens{:}));
        ids    = data(:, 1);
        values = data(:, 2);
    else
        ids = []; values = [];
    end
end