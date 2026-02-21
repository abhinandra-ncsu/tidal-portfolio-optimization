%% UTide Tidal Harmonic Analysis
% =========================================================================
% Step 2 of the UTide workflow: Runs tidal harmonic analysis on velocity
% data prepared by run_energy_pipeline.py (tidal mode).
%
% Input:
%   - outputs/<region>/utide_input/{lat}_{lon}.mat files
%     Each file contains: formatted_times, uin, vin, lat
%
% Output:
%   - outputs/<region>/utide_output/{lat}_{lon}.mat files
%     Each file contains: u_recon, v_recon
%   - outputs/<region>/utide_diagnostics/{lat}_{lon}.mat files
%     Each file contains: coef (full UTide harmonic analysis struct)
%
% Requirements:
%   - UTide library must be available. Copy UTide files to lib/utide/
%     or ensure they are already on the MATLAB path.
%
% Usage:
%   1. Run energy pipeline (tidal mode): python scripts/run_energy_pipeline.py
%   2. Run this script in MATLAB: run('scripts/run_utide_analysis')
%   3. Re-run energy pipeline: python scripts/run_energy_pipeline.py
% =========================================================================

%% Locate project root (parent of scripts/ directory)
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);

%% Configuration — change REGION to match your data
region = 'North_Carolina';

inputDir  = fullfile(projectRoot, 'outputs', region, 'utide_input');
outputDir = fullfile(projectRoot, 'outputs', region, 'utide_output');
diagDir   = fullfile(projectRoot, 'outputs', region, 'utide_diagnostics');

%% Add UTide to path (if not already added)
if ~exist('ut_solv', 'file')
    utideLib = fullfile(projectRoot, 'lib', 'utide');
    if exist(utideLib, 'dir')
        addpath(utideLib);
        fprintf('Added %s to MATLAB path.\n', utideLib);
    else
        error(['UTide library not found. Please copy UTide files to lib/utide/\n' ...
               'Download from: https://www.mathworks.com/matlabcentral/fileexchange/46523']);
    end
end

%% Create output directories if they don't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
    fprintf('Created output directory: %s\n', outputDir);
end
if ~exist(diagDir, 'dir')
    mkdir(diagDir);
    fprintf('Created diagnostics directory: %s\n', diagDir);
end

%% Get list of input files
files = dir(fullfile(inputDir, '*.mat'));
numFiles = length(files);

if numFiles == 0
    error('No .mat files found in %s.\nRun: python scripts/run_energy_pipeline.py', inputDir);
end

fprintf('\n========================================\n');
fprintf('UTide Tidal Harmonic Analysis\n');
fprintf('========================================\n');
fprintf('Region:          %s\n', region);
fprintf('Input directory:  %s\n', inputDir);
fprintf('Output directory: %s\n', outputDir);
fprintf('Files to process: %d\n\n', numFiles);

%% Process each file
successCount = 0;
failCount = 0;
startTime = tic;

for k = 1:numFiles
    inputFilePath = fullfile(inputDir, files(k).name);

    try
        % Load the prepared data
        data = load(inputFilePath);

        % Convert formatted time strings to MATLAB datenums
        % Format: "dd-mmm-yyyy HH:MM:SS"
        tin = datenum(data.formatted_times, 'dd-mmm-yyyy HH:MM:SS');

        % Run UTide harmonic analysis
        % 'auto' lets UTide automatically select tidal constituents
        % based on the record length
        coef = ut_solv(tin, data.uin, data.vin, data.lat, 'auto');

        % Reconstruct tidal velocities from the harmonic constituents
        [u_recon, v_recon] = ut_reconstr(tin, coef);

        % Save reconstructed velocities
        [~, baseFileName, ~] = fileparts(files(k).name);
        outputFilePath = fullfile(outputDir, sprintf('%s.mat', baseFileName));
        save(outputFilePath, 'u_recon', 'v_recon');

        % Save full UTide coefficient struct for diagnostics
        diagFilePath = fullfile(diagDir, sprintf('%s.mat', baseFileName));
        save(diagFilePath, 'coef');

        successCount = successCount + 1;

        % Progress update every 50 files or at the end
        if mod(k, 50) == 0 || k == numFiles
            fprintf('Processed %d/%d files (%.1f%%)\n', k, numFiles, 100*k/numFiles);
        end

    catch ME
        failCount = failCount + 1;
        fprintf('ERROR processing %s: %s\n', files(k).name, ME.message);
    end
end

%% Summary
elapsedTime = toc(startTime);
fprintf('\n========================================\n');
fprintf('Processing Complete\n');
fprintf('========================================\n');
fprintf('Successful: %d\n', successCount);
fprintf('Failed:     %d\n', failCount);
fprintf('Elapsed time: %.1f seconds (%.2f sec/file)\n', elapsedTime, elapsedTime/numFiles);
fprintf('Output saved to: %s/\n', outputDir);
fprintf('Diagnostics saved to: %s/\n', diagDir);
fprintf('\nNext step: python scripts/run_energy_pipeline.py\n');
