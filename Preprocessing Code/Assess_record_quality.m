clc; clear; close all;

%% Locate Chunk Files
chunk_files = dir('preprocessed_ecg_data_chunk_*.mat');
if isempty(chunk_files)
    error('No chunk files found (preprocessed_ecg_data_chunk_*.mat) in this folder.');
end

%Sort chunk_files in ascending numeric order based on the chunk index
numeric_indices = zeros(length(chunk_files),1);
for i = 1:length(chunk_files)
    % Extract the number from the filename using a regular expression
    tokens = regexp(chunk_files(i).name, 'preprocessed_ecg_data_chunk_(\d+)\.mat', 'tokens');
    if ~isempty(tokens)
        numeric_indices(i) = str2double(tokens{1}{1});
    else
        % If the pattern doesn't match, set to Inf to push it to the end
        numeric_indices(i) = Inf;
    end
end

[~, sorted_idx] = sort(numeric_indices);
chunk_files = chunk_files(sorted_idx);

output_folder = 'filtered_chunks';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

dropped_output_folder = 'dropped_records';
if ~exist(dropped_output_folder, 'dir')
    mkdir(dropped_output_folder);
end

%% Thresholds for Outlier Detection

% Baseline drift
baseline_min   = 0.03;  
baseline_max   = 0.97; 

% Spike Detection
spike_threshold = 0.5;  % No consecutive sample jump larger than this

% Flatline Detection
flatline_min_range = 0.05;
flatline_std_threshold = 0.001;

% Amplitude Range
max_allowed_range = 1.1;          % If range > 1.1, signal is suspicious

% Standard Deviation
min_std   = 0.005;
max_std   = 0.25;

% Sliding Window Parameters for Flatline Detection
window_size = 250; %in samples
overlap = 125; 

%R Peak Detection

fs = 500; 
signal_duration = 10;
total_samples = fs * signal_duration; % 5000

hr_min = 40;       
hr_max = 200;   

%% Initialize Counters and Plotting Parameters for Each Outlier Type

outlier_types = { ...
    'baseline_drift', ...
    'large_spike', ...
    'flatline', ...
    'amp_range', ...
    'std_low', ...
    'std_high', ...
    'invalid_hr'};

% Define how often to plot each outlier type
plot_frequency = containers.Map( ...
    {'baseline_drift', 'large_spike', 'flatline', 'std_high', 'invalid_hr'}, ...
    {2, 25, 2, 10, 20}); 

outlier_counters = containers.Map(outlier_types, num2cell(zeros(1, length(outlier_types))));

dropped_records = struct('record_name', {}, 'drop_reason', {}, 'signal', {}, 'problematic_lead', {});

%% Initialize Plotting Structures for Each Outlier Type
plots_per_figure = 16; % e.g., 16 subplots per figure (4x4)
subplot_rows = 4;
subplot_cols = 4;

plot_structures = struct();
for i = 1:length(outlier_types)
    type = outlier_types{i};
    if isKey(plot_frequency, type)
        plot_structures.(type).figure_count = 1;
        plot_structures.(type).current_plot = 1;
        plot_structures.(type).figure_handle = figure('Name', ...
            ['Dropped ECG Leads - ', type, ' - Figure 1'], 'Color', 'w');
        set(plot_structures.(type).figure_handle, 'Position', [100, 100, 1600, 1200]);
    end
end

%% Process Each Chunk File
for file_idx = 1:length(chunk_files)
    chunk_filename = chunk_files(file_idx).name;
    fprintf('\n=====================================\n');
    fprintf('Processing chunk file: %s\n', chunk_filename);
    loaded_data = load(chunk_filename, 'preprocessed_data');
    if ~isfield(loaded_data, 'preprocessed_data')
        fprintf('  -> No "preprocessed_data" found in %s. Skipping.\n', chunk_filename);
        continue;
    end
    
    preprocessed_data = loaded_data.preprocessed_data;
    record_names_in_chunk = fieldnames(preprocessed_data);
    
    % Initialize structure to store good records
    filtered_data = struct();
    
    good_count = 0;
    dropped_count = 0;  % local counter for this chunk
    
   
    for rn_idx = 1:length(record_names_in_chunk)
        rec_name = record_names_in_chunk{rn_idx};
        record_struct = preprocessed_data.(rec_name);
        
        % 12-lead ECG data
        signal_12lead = record_struct.signal;  % 12 x 5000
        
        % Evaluate if this record is "good" or not
        [is_good, drop_reason, outlier_type, problematic_lead] = is_good_ecg(...
            signal_12lead, ...
            baseline_min, baseline_max, ...
            spike_threshold, ...
            flatline_min_range, flatline_std_threshold, ...
            window_size, overlap, max_allowed_range, ...
            min_std, max_std, ...
            fs, hr_min, hr_max);
        
        if is_good
            % Keep record
            filtered_data.(rec_name) = record_struct;
            good_count = good_count + 1;
        else
            % Drop record
            dropped_count = dropped_count + 1;
            
            dropped_records(end+1).record_name = rec_name; %#ok<AGROW>
            dropped_records(end).drop_reason = drop_reason;
            dropped_records(end).signal = signal_12lead;
            dropped_records(end).problematic_lead = problematic_lead; %#ok<AGROW>
            
            % Update outlier counters
            if isKey(outlier_counters, outlier_type)
                outlier_counters(outlier_type) = outlier_counters(outlier_type) + 1;
            else
                outlier_types{end+1} = outlier_type; %#ok<AGROW>
                outlier_counters(outlier_type) = 1;
            end
            
            % Determine if we should plot this outlier
            if isKey(plot_frequency, outlier_type)
                current_count = outlier_counters(outlier_type);
                N = plot_frequency(outlier_type);
                should_plot = false;
                
                if current_count == 1
                    % Always plot the first occurrence
                    should_plot = true;
                elseif mod(current_count, N) == 0
                    % Plot every N-th occurrence
                    should_plot = true;
                end
                
                if should_plot
                    type = outlier_type;
                    fig_struct = plot_structures.(type);
                  
                    if fig_struct.current_plot > plots_per_figure
                        fig_struct.figure_count = fig_struct.figure_count + 1;
                        fig_struct.figure_handle = figure('Name', ...
                            ['Dropped ECG Leads - ', type, ' - Figure ', num2str(fig_struct.figure_count)], ...
                            'Color', 'w');
                        set(fig_struct.figure_handle, 'Position', [100, 100, 1600, 1200]);
                        fig_struct.current_plot = 1;
                    end
                    
                    % Create subplot
                    ax = subplot(subplot_rows, subplot_cols, ...
                        fig_struct.current_plot, 'Parent', fig_struct.figure_handle);
                    
                    % Plot the problematic lead
                    plot_dropped_ecg_example(signal_12lead, rec_name, ...
                        drop_reason, outlier_type, problematic_lead, ax, fs);
                    
                    fig_struct.current_plot = fig_struct.current_plot + 1;
                    plot_structures.(type) = fig_struct;
                end
            end
        end
    end
    
    %Save good records to new file (if any)
    if ~isempty(fieldnames(filtered_data))
        [~, base_name] = fileparts(chunk_filename);
        output_filename = fullfile(output_folder, [base_name, '_filtered.mat']);
        
        save(output_filename, 'filtered_data', '-v7.3');
        fprintf('Saved %d good records (dropped %d) to:\n  %s\n', ...
            good_count, dropped_count, output_filename);
    else
        fprintf('All records in %s were dropped! No output file saved.\n', chunk_filename);
    end
end

%Save all dropped records to a single file
dropped_output_filename = fullfile(dropped_output_folder, 'all_dropped_records.mat');
save(dropped_output_filename, 'dropped_records', '-v7.3');
fprintf('\nAll dropped records saved to:\n  %s\n', dropped_output_filename);
fprintf('\nDone filtering all chunk files.\n');

%% HELPER FUNCTION: Evaluate a 12-lead record
function [is_good, reason, outlier_type, problematic_lead] = is_good_ecg(ecg_data, ...
    baseline_min, baseline_max, ...
    spike_threshold, ...
    flatline_min_range, flatline_std_threshold, ...
    window_size, overlap, max_allowed_range, ...
    min_std, max_std, ...
    fs, hr_min, hr_max)

    is_good = true;
    reason  = '';
    outlier_type = '';
    problematic_lead = NaN;

    [num_leads, sig_length] = size(ecg_data);
    for lead_idx = 1:num_leads
        sig = ecg_data(lead_idx, :);

        % 1.1) Baseline drift check
        mval = mean(sig);
        if (mval < baseline_min || mval > baseline_max)
            is_good = false;
            reason = sprintf('Lead %d baseline drift (mean=%.3f)', lead_idx, mval);
            outlier_type = 'baseline_drift';
            problematic_lead = lead_idx;
            return;
        end

        % 1.2) Large spikes check
        diffs = abs(diff(sig));
        if max(diffs) > spike_threshold
            is_good = false;
            reason = sprintf('Lead %d has large spike (> %.2f)', lead_idx, spike_threshold);
            outlier_type = 'large_spike';
            problematic_lead = lead_idx;
            return;
        end

        % 1.3) Flatline check
        lead_range = max(sig) - min(sig);
        if lead_range < flatline_min_range
            if is_flatline(sig, window_size, overlap, flatline_std_threshold)
                is_good = false;
                reason = sprintf('Lead %d is nearly flat (range=%.4f)', lead_idx, lead_range);
                outlier_type = 'flatline';
                problematic_lead = lead_idx;
                return;
            end
        end

        % 1.4) Amplitude range check
        if lead_range > max_allowed_range
            is_good = false;
            reason = sprintf('Lead %d amplitude range > %.2f', lead_idx, max_allowed_range);
            outlier_type = 'amp_range';
            problematic_lead = lead_idx;
            return;
        end

        % 1.5) Standard deviation check
        sig_std = std(double(sig));  
        if sig_std < min_std
            is_good = false;
            reason = sprintf('Lead %d std dev < %.3f', lead_idx, min_std);
            outlier_type = 'std_low';
            problematic_lead = lead_idx;
            return;
        elseif sig_std > max_std
            is_good = false;
            reason = sprintf('Lead %d std dev > %.3f', lead_idx, max_std);
            outlier_type = 'std_high';
            problematic_lead = lead_idx;
            return;
        end
    end

    %R Peak Detection and Heart Rate Validation
    % Use Lead II (index 2) as a representative lead
    representative_lead = ecg_data(2, :);

    r_peaks = detect_r_peaks(representative_lead, fs);

    % Calculate heart rate
    num_beats = length(r_peaks);
    duration_minutes = sig_length / fs / 60; 
    hr = num_beats / duration_minutes;

    % Validate HR
    if hr < hr_min || hr > hr_max
        is_good = false;
        reason = sprintf('Heart rate out of bounds (HR=%.1f bpm)', hr);
        outlier_type = 'invalid_hr';
        problematic_lead = 2; % lead II
        return;
    end
end

%% HELPER FUNCTION: Detect R Peaks in an ECG  (as seen in class)
function r_peaks = detect_r_peaks(ecg_signal, fs)
    % 1. Bandpass filter (5-15 Hz)
    low_cut = 5;
    high_cut = 15;
    [b, a] = butter(2, [low_cut high_cut]/(fs/2), 'bandpass');
    filtered_ecg = filtfilt(b, a, ecg_signal);

    diff_ecg = diff(filtered_ecg);

    squared_ecg = diff_ecg .^ 2;

    window_size = round(0.150 * fs); 
    integrated_ecg = movmean(squared_ecg, window_size);

    peak_scale_factor = 0.35
    peak_max = max(integrated_ecg);
    
    if peak_max > 1e-10 
        min_peak_height = peak_max * peak_scale_factor;
    else
        % Fallback if the signal is very small or near zero
        min_peak_height = 0;  
    end
    min_distance = round(0.250 * fs); 

    [~, locs] = findpeaks(integrated_ecg, 'MinPeakHeight', min_peak_height, ...
        'MinPeakDistance', min_distance);

    r_peaks = [];
    search_window = round(0.150 * fs);

    for i = 1:length(locs)
        window_start = max(locs(i)-search_window, 1);
        window_end   = min(locs(i)+search_window, length(filtered_ecg));
        [~, local_max] = max(filtered_ecg(window_start:window_end));
        r_peak = window_start - 1 + local_max;
        r_peaks = [r_peaks; r_peak];
    end

    r_peaks = unique(r_peaks);
end

%% HELPER FUNCTION: Determine if a signal is flatline by sliding window
function is_flat = is_flatline(signal, window_size, overlap, std_threshold)
    num_samples = length(signal);
    step = window_size - overlap;
    num_windows = floor((num_samples - overlap) / step);
    is_flat = true; 

    for w = 1:num_windows
        start_idx = (w-1)*step + 1;
        end_idx   = start_idx + window_size - 1;
        if end_idx > num_samples
            end_idx = num_samples;
        end
        window = signal(start_idx:end_idx);
        window_std = std(double(window));
        if window_std > std_threshold
            is_flat = false;
            return;
        end
    end
end

%% HELPER FUNCTION: Plot a dropped record's problematic lead

function plot_dropped_ecg_example(signal_12lead, rec_name, drop_reason, ...
    outlier_type, problematic_lead, ax, fs)

    % Extract the problematic lead
    lead_signal = signal_12lead(problematic_lead, :);
    
    plot(ax, lead_signal, 'b-');
    title(ax, sprintf('Record: %s\nLead: %d\nType: %s', ...
        rec_name, problematic_lead, outlier_type), ...
        'FontSize', 10, 'Interpreter', 'none');
    xlabel(ax, 'Sample');
    ylabel(ax, 'Amplitude');
    ylim(ax, [min(signal_12lead(:)) max(signal_12lead(:))]);
    grid(ax, 'on');
    
    % If outlier is invalid HR, show R peaks
    if strcmp(outlier_type, 'invalid_hr')
        r_peaks = detect_r_peaks(lead_signal, fs);
        hold(ax, 'on');
        plot(ax, r_peaks, lead_signal(r_peaks), 'ro', 'MarkerFaceColor', 'r');
        hold(ax, 'off');
    end
    
    drawnow;
end
