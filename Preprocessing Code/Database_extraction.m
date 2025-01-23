% Extraction of the data from the dataset
% Data is stored in a structure called extracted_data.

% Set the dataset root folder
root_folder = 'E:\Polimi\Biomedical Engineering\Semester 3\Lab\Assignment\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords';

first_level_dirs = dir(fullfile(root_folder, '*'));
first_level_dirs = first_level_dirs([first_level_dirs.isdir] & ~startsWith({first_level_dirs.name}, '.'));

extracted_data = struct();
record_count = 0;

for i = 1:length(first_level_dirs)
    second_level_path = fullfile(root_folder, first_level_dirs(i).name);
    second_level_dirs = dir(fullfile(second_level_path, '*'));
    second_level_dirs = second_level_dirs([second_level_dirs.isdir] & ~startsWith({second_level_dirs.name}, '.'));

    for j = 1:length(second_level_dirs)
        record_path = fullfile(second_level_path, second_level_dirs(j).name);

        % Read the RECORDS file
        records_file = fullfile(record_path, 'RECORDS');
        if exist(records_file, 'file')
            fileID = fopen(records_file, 'r');
            record_names = textscan(fileID, '%s');
            fclose(fileID);
            record_names = record_names{1}; % Extract cell array of record names

            for k = 1:length(record_names)
                record_base = fullfile(record_path, record_names{k});
                mat_file_path = strcat(record_base, '.mat');
                hea_file_path = strcat(record_base, '.hea');

                try
                    % Check if .mat file exists
                    if exist(mat_file_path, 'file')
                        % Load the .mat file
                        mat_data = load(mat_file_path);

                        field_names = fieldnames(mat_data);
                        ecg_data = mat_data.(field_names{1}); % Access the signal data

                        % Read the .hea file for metadata
                        if exist(hea_file_path, 'file')
                            header = fileread(hea_file_path);
                        else
                            header = 'Header file not found.';
                        end

                        % Store the data in the structure
                        record_count = record_count + 1;
                        extracted_data.(sprintf('record_%d', record_count)).signal = ecg_data;
                        extracted_data.(sprintf('record_%d', record_count)).header = header;

                        disp(['Processed: ', record_base]);
                    else
                        warning(['MAT file not found: ', mat_file_path]);
                    end
                catch ME
                    warning(['Error processing ', record_base, ': ', ME.message]);
                end
            end
        else
            warning(['RECORDS file not found in folder: ', record_path]);
        end
    end
end

% Save extracted data to a .mat file using MAT-file version 7.3
save('extracted_ecg_data.mat', 'extracted_data', '-v7.3');
disp(['Total records processed: ', num2str(record_count)]);