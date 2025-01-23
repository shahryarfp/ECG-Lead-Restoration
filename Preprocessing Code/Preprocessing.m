clc; clearvars -except extracted_data; close all;
%%
if ~exist('extracted_data', 'var')
    load('extracted_ecg_data.mat');
end
record_names = fieldnames(extracted_data);

%% Parameters
tStart   = tic;  % Start timing
fs       = 500; 
low_cutoff  = 0.05;   % for bandpass (Hz)
high_cutoff = 150;    % for bandpass (Hz)
highpass_cutoff = 0.5; % for baseline removal (Hz)
lambda       = 1e7;   % for arPLS
max_iter     = 10;    % for arPLS

% Design your filters once (butter(2)):
[b_bandpass, a_bandpass]   = butter(2, [low_cutoff, high_cutoff]/(fs/2), 'bandpass');
[b_highpass, a_highpass]   = butter(2, highpass_cutoff/(fs/2), 'high');

% Chunking
chunk_size  = 1000;  % Number of records per chunk
num_records = length(record_names);
num_chunks  = ceil(num_records / chunk_size);

% Process each chunk
for chunk_idx = 1:num_chunks
    
    % Calculate start/end indices for this chunk
    chunk_start = (chunk_idx - 1) * chunk_size + 1;
    chunk_end   = min(chunk_idx * chunk_size, num_records);
    
    % Prepare a struct to hold this chunk’s results
    preprocessed_data = struct();
    
    % Get the record names for this chunk
    chunk_record_names = record_names(chunk_start:chunk_end);

    % Iterate through the records in the current chunk
    for i = 1:length(chunk_record_names)
        current_record_name = chunk_record_names{i};
        record  = extracted_data.(current_record_name);
        signal  = record.signal;  % 12-lead ECG data

        % 1) Skip records with fewer or more than 12 leads
        if size(signal, 1) ~= 12
            disp(['Skipping record ', record_names{i}, ': Incomplete leads!']);
            continue;
        end
        
        % 2) Check for sufficient signal length (at least 10 s @ 500 Hz)
        if size(signal, 2) ~= 5000
            disp(['Skipping record ', record_names{i}, ': Incomplete Signal length!']);
            continue;
        end

        % 3) Check for NaN values in the signal
        if any(isnan(signal(:)))
            warning(['Skipping entire record ', current_record_name, ' due to NaN values.']);
            continue;  % Skip processing this record and move to the next one
        end
        
        % Preallocate single-precision array for preprocessed signals
        preprocessed_signals = zeros(size(signal), 'single');
        
        % Flag to determine if the entire record should be skipped
        skip_record = false;
        
        % Process each lead
        for lead_idx = 1:size(signal, 1)
            original_signal = signal(lead_idx, :);
            
            % Step 1: Bandpass Filtering
            if length(original_signal) > 12
                bandpassed_signal = filtfilt(b_bandpass, a_bandpass, original_signal);
            else
                warning(['Signal in record ', current_record_name, ...
                         ' lead ', num2str(lead_idx), ' too short for filtering. Skipping.']);
                bandpassed_signal = original_signal;
            end
            
            % Step 2: Baseline Removal (High-Pass)
            baseline_removed_signal = filtfilt(b_highpass, a_highpass, bandpassed_signal);
    
            % Step 3: arPLS (baseline correction)
            corrected_signal = remove_baseline_arpls(baseline_removed_signal', lambda, max_iter);
            
            % Check if corrected_signal has finite values
            if ~all(isfinite(corrected_signal))
                warning(['arPLS produced non-finite values in record ', current_record_name, ...
                         ' lead ', num2str(lead_idx), '. Skipping this record.']);
                skip_record = true;
                break;  % Exit the lead processing loop to skip the record
            end

            % Step 4: convert to mv and check the boundary
            adc_to_uV = 4.88;         % 4.88 microvolts per A/D bit
            uV_to_mV  = 1 / 1000;     % Convert microvolts to millivolts
            lowerBound = -10;
            upperBound =  10;
            signal_mV = corrected_signal * adc_to_uV * uV_to_mV;
            if any(signal_mV < lowerBound) || any(signal_mV > upperBound)
                warning(['Signal in record ', current_record_name, ...
                     ' lead ', num2str(lead_idx), ...
                     ' is out of ±10 mV range.']);
                skip_record = true;
                break;
            end

            % Step 5: Wavelet Denoising
            try
                corrected_signal = wavelet_denoise(signal_mV);
            catch
                warning(['Wavelet denoising failed for record ', current_record_name, ...
                         ' lead ', num2str(lead_idx), '. Skipping this record.']);
                skip_record = true;
                break;
            end

            % Step 6: Min-Max Normalization
            min_val = min(corrected_signal);
            max_val = max(corrected_signal);
            if max_val - min_val > 0
                normalized_signal = (corrected_signal - min_val) / (max_val - min_val);
            else
                warning(['Signal in record ', current_record_name, ...
                         ' lead ', num2str(lead_idx), ' has no variation. Skipping normalization.']);
                normalized_signal = corrected_signal;
            end
            
            % Store as single precision
            preprocessed_signals(lead_idx, :) = single(normalized_signal);
        end
        
        if skip_record
            warning(['Skipping entire record ', current_record_name, ' due to processing issues.']);
            continue;  % Skip saving this record and proceed to the next one
        end
        
        % Save signals in the chunk's struct
        preprocessed_data.(current_record_name).signal = preprocessed_signals;
        preprocessed_data.(current_record_name).header = record.header;  % keep metadata

        disp(['Preprocessed record: ', current_record_name]);
    end

    
    % Save the chunk to a separate .mat file (using a unique name per chunk)
    chunk_filename = sprintf('preprocessed_ecg_data_chunk_%d.mat', chunk_idx);
    save(chunk_filename, 'preprocessed_data', '-v7.3');
    
    % Clear the struct to free memory before moving to the next chunk
    clear preprocessed_data;
    
    disp(['Completed chunk ', num2str(chunk_idx), '/', num2str(num_chunks), ...
          ' (records ', num2str(chunk_start), ' to ', num2str(chunk_end), ').']);
end

% You can combine the chunk files later if needed, or process them individually.

% Timing
elapsedTime = toc(tStart);
disp(['Total Elapsed time: ', num2str(elapsedTime), ' seconds.']);

%% arPLS Function Definition
function corrected_signal = remove_baseline_arpls(signal, lambda, max_iter)
    % Removes baseline wander using Adaptive Reweighted Penalized Least Squares (arPLS)
    % Parameters:
    %   signal    : 1D array of the signal
    %   lambda    : Regularization parameter (higher = smoother baseline)
    %   max_iter  : Maximum number of iterations

    N = length(signal);
    D = diff(speye(N), 2);   % Second-order difference matrix
    H = lambda * (D' * D); 

    w = ones(N, 1);          % Weights (updated iteratively)
    for iter = 1:max_iter
        W = spdiags(w, 0, N, N);
        C = W + H;
        baseline = C \ (w .* signal);

        % Update weights
        residuals = signal - baseline;
        neg_idx   = residuals < 0;
        stdev_res = std(residuals);
        % Weighted differently: if residual < 0, set w=1; else exponential
        w = exp(-abs(residuals) / stdev_res);
        w(neg_idx) = 1;
    end

    corrected_signal = signal - baseline;
end

function denoised_signal = wavelet_denoise(signal)
    waveletName = 'sym8';  % Daubechies 6 is effective for ECG
    maxLevel = 5;
    
    % Perform wavelet denoising using wdenoise
    denoised_signal = wdenoise(signal, maxLevel, ...
        'Wavelet', waveletName, ...
        'DenoisingMethod', 'SURE', ...
        'ThresholdRule', 'Soft', ...
        'NoiseEstimate', 'LevelDependent');
    
    % Ensure the denoised signal has the same length as the original signal
    denoised_signal = denoised_signal(1:length(signal));
end
