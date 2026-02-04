% Script: Environmental Factors Time Series Analysis
% Plot time series data of environmental factors from Excel file
% Author: Scientific Analysis System
% Date: February 2026

clear all; close all; clc;

% Excel filename
filename = 'SickTree.xlsx';

% Sheet names to read
sheets = {'Chuẩn 1', 'N.Chuẩn 3 - 2', 'N.Chuẩn 2 - 1', 'Not A - 1'};
sheet_labels = {'Standard 1', 'Standard 3-2', 'Standard 2-1', 'Not A-1'};

% Scientific plot settings
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultTextFontName', 'Arial');
set(0, 'DefaultLineLineWidth', 1.5);

% Data column names (excluding first column - sample number)
factor_names = {'R', 'EC', 'T', 'H', 'CO2', 'LUX', 'Sound', 'Soil'};
factor_full_names = {
    'Plant Tissue Resistance', ...
    'Electrical Conductivity', ...
    'Air Temperature', ...
    'Air Humidity', ...
    'CO2 Concentration', ...
    'Light Intensity', ...
    'Sound Intensity', ...
    'Soil Moisture'
};
factor_units = {'kOhm', 'uS', 'degC', '%', 'ppm', 'lx', 'dB', '%'};

% Color scheme for each group (scientifically distinguishable)
colors = [
    0.00, 0.45, 0.74;  % Deep blue (Standard 1)
    0.85, 0.33, 0.10;  % Dark orange (Standard 3-2)
    0.93, 0.69, 0.13;  % Gold (Standard 2-1)
    0.49, 0.18, 0.56   % Purple (Not A-1)
];

% Read data from all sheets
data_all = cell(1, length(sheets));
for i = 1:length(sheets)
    try
        % Read data as matrix directly
        [num_data, ~, ~] = xlsread(filename, sheets{i});
        
        % Extract first 9 columns (sample number + 8 factors)
        % Skip first row which is header
        if size(num_data, 2) >= 9
            data_all{i} = num_data(:, 1:9);
            fprintf('Successfully read sheet "%s": %d rows x %d columns\n', sheets{i}, size(num_data, 1), 9);
        else
            data_all{i} = [];
            fprintf('Warning: Sheet "%s" has only %d columns\n', sheets{i}, size(num_data, 2));
        end
    catch ME
        fprintf('Error reading sheet "%s": %s\n', sheets{i}, ME.message);
        data_all{i} = [];
    end
end

% Create figure with scientific layout
fig = figure('Position', [100, 100, 1600, 1000], 'Color', 'w', 'PaperPositionMode', 'auto');
sgtitle('Time Series Analysis of Environmental Factors', ...
    'FontSize', 18, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.1], 'FontName', 'Arial');

% Create 8 subplots for 8 factors
for factor_idx = 1:8
    subplot(3, 3, factor_idx);
    hold on;
    grid on;
    box on;
    
    % Plot data for each group
    legend_entries = {};
    for sheet_idx = 1:length(sheets)
        if ~isempty(data_all{sheet_idx})
            data = data_all{sheet_idx};
            
            % Extract corresponding column (factor_idx + 1 since column 1 is sample number)
            if size(data, 2) > factor_idx
                y_data = data(:, factor_idx + 1);
                
                % Remove NaN values
                valid_idx = ~isnan(y_data);
                y_data = y_data(valid_idx);
                x_data = find(valid_idx);
                
                % Plot with appropriate color and line width
                if ~isempty(x_data) && ~isempty(y_data)
                    plot(x_data, y_data, 'LineWidth', 1.8, 'Color', colors(sheet_idx, :), ...
                        'DisplayName', sheet_labels{sheet_idx});
                    legend_entries{end+1} = sheet_labels{sheet_idx};
                end
            end
        end
    end
    
    % Format subplot with scientific standards
    xlabel('Measurement Number', 'FontSize', 11, 'FontWeight', 'bold', 'FontName', 'Arial');
    ylabel(sprintf('%s (%s)', factor_names{factor_idx}, factor_units{factor_idx}), ...
        'FontSize', 11, 'FontWeight', 'bold', 'FontName', 'Arial');
    title(factor_full_names{factor_idx}, ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.2 0.2 0.2], 'FontName', 'Arial');
    
    % Add legend
    if factor_idx == 1
        lgd = legend('Location', 'best', 'FontSize', 9, 'FontName', 'Arial');
        lgd.Box = 'on';
        lgd.EdgeColor = [0.7 0.7 0.7];
    end
    
    % Customize axes for scientific publication
    ax = gca;
    ax.FontSize = 9;
    ax.FontName = 'Arial';
    ax.LineWidth = 1.0;
    ax.GridAlpha = 0.2;
    ax.GridLineStyle = ':';
    ax.Box = 'on';
    ax.TickDir = 'out';
    ax.TickLength = [0.01 0.01];
    
    hold off;
end

% Subplot 9: Display comprehensive legend and information
subplot(3, 3, 9);
axis([0 1 0 1]);
axis off;

% Create comprehensive legend
text_y = 0.88;
text(0.08, text_y, 'Data Groups:', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Arial');
for i = 1:length(sheet_labels)
    text_y = text_y - 0.13;
    rectangle('Position', [0.08, text_y-0.02, 0.06, 0.06], ...
        'FaceColor', colors(i, :), 'EdgeColor', 'k', 'LineWidth', 0.8);
    text(0.16, text_y, sheet_labels{i}, 'FontSize', 10, 'FontWeight', 'normal', 'FontName', 'Arial');
end

% Additional information
text_y = text_y - 0.16;
text(0.08, text_y, 'Experiment Info:', 'FontSize', 10, 'FontWeight', 'bold', 'FontName', 'Arial');
text_y = text_y - 0.10;
text(0.08, text_y, sprintf('Total groups: %d', length(sheets)), 'FontSize', 9, 'FontName', 'Arial');
text_y = text_y - 0.09;
text(0.08, text_y, 'Continuous monitoring', 'FontSize', 9, 'FontName', 'Arial');
text_y = text_y - 0.09;
text(0.08, text_y, '8 environmental factors', 'FontSize', 9, 'FontName', 'Arial');
text_y = text_y - 0.09;
text(0.08, text_y, 'Resolution: 300 DPI', 'FontSize', 9, 'FontName', 'Arial');

% Create plots directory if it doesn't exist
if ~exist('plots', 'dir')
    mkdir('plots');
end

% Save figure
print(fig, 'plots/all_factors_plot.png', '-dpng', '-r300');
fprintf('\n[OK] Saved combined plot: plots/all_factors_plot.png\n');

% Create individual figure for each factor (detailed view)
for factor_idx = 1:8
    fig2 = figure('Position', [100, 100, 1200, 600], 'Color', 'w', 'PaperPositionMode', 'auto');
    hold on;
    grid on;
    box on;
    
    % Plot data for each group
    for sheet_idx = 1:length(sheets)
        if ~isempty(data_all{sheet_idx})
            data = data_all{sheet_idx};
            
            if size(data, 2) > factor_idx
                y_data = data(:, factor_idx + 1);
                
                % Remove NaN values
                valid_idx = ~isnan(y_data);
                y_data = y_data(valid_idx);
                x_data = find(valid_idx);
                
                if ~isempty(x_data) && ~isempty(y_data)
                    plot(x_data, y_data, 'LineWidth', 2.0, 'Color', colors(sheet_idx, :), ...
                        'DisplayName', sheet_labels{sheet_idx}, 'Marker', 'none');
                end
            end
        end
    end
    
    % Format for scientific publication
    xlabel('Measurement Number', 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Arial');
    ylabel(sprintf('%s (%s)', factor_names{factor_idx}, factor_units{factor_idx}), ...
        'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Arial');
    title(sprintf('%s - Time Series Analysis', factor_full_names{factor_idx}), ...
        'FontSize', 15, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.1], 'FontName', 'Arial');
    
    lgd = legend('Location', 'best', 'FontSize', 11, 'FontName', 'Arial');
    lgd.Box = 'on';
    lgd.EdgeColor = [0.7 0.7 0.7];
    
    ax = gca;
    ax.FontSize = 11;
    ax.FontName = 'Arial';
    ax.LineWidth = 1.2;
    ax.GridAlpha = 0.25;
    ax.GridLineStyle = ':';
    ax.Box = 'on';
    ax.TickDir = 'out';
    
    hold off;
    
    % Save individual plot
    filename_save = sprintf('plots/factor_%s_plot.png', factor_names{factor_idx});
    print(fig2, filename_save, '-dpng', '-r300');
    fprintf('[OK] Saved %s plot: %s\n', factor_names{factor_idx}, filename_save);
end

fprintf('\n=== COMPLETE === Generated %d plots successfully.\n', 1 + length(factor_names));
