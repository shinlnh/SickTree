% Create visualizations for structured MLP output
% This script generates comprehensive plots showing model predictions with
% prominent conditions, health status, and recommendations

clear; clc; close all;

%% Define test scenarios based on the classification table
scenarios = {
    struct('name', 'Healthy Baseline', ...
           'label', 'N.Chuẩn 3 - 2', ...
           'R', 47.4, 'EC', 26.45, 'T', 29.4, 'H', 88, ...
           'CO2', 910, 'Lux', 85, 'Sound', 45.2, 'Soil', 71, ...
           'status', 'Không bệnh (baseline/healthy)', ...
           'group', 'Khỏe mạnh', ...
           'recommendation', 'Duy trì điều kiện hiện tại', ...
           'probs', [0.01, 0.02, 0.09, 0.01, 0.01, 0.01, 0.06, 0.05, 99.71, 0.01, 0.01, 0.02]), ...
    
    struct('name', 'Light Stress', ...
           'label', 'Chuẩn 1', ...
           'R', 96, 'EC', 96, 'T', 29, 'H', 70, ...
           'CO2', 800, 'Lux', 101, 'Sound', 45, 'Soil', 65, ...
           'status', 'Thừa một phần sáng', ...
           'group', 'Stress ánh sáng', ...
           'recommendation', 'Giảm ánh sáng nhẹ - tráng rủi ro khô', ...
           'probs', [100, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]), ...
    
    struct('name', 'Insufficient Irrigation', ...
           'label', 'N.Chuẩn 1 - 1', ...
           'R', 210, 'EC', 15, 'T', 32, 'H', 63, ...
           'CO2', 600, 'Lux', 101, 'Sound', 45, 'Soil', 62, ...
           'status', 'Thừa sáng thiếu ẩm nhẹ', ...
           'group', 'Thiếu nước/ẩm', ...
           'recommendation', 'Tăng độ ẩm, giảm ánh sáng', ...
           'probs', [0.05, 99.85, 0.01, 0.01, 0.01, 0.02, 0.03, 0.00, 0.00, 0.00, 0.00, 0.02]), ...
    
    struct('name', 'Excessive Moisture', ...
           'label', 'N.Chuẩn 1 - 2', ...
           'R', 86, 'EC', 19, 'T', 28, 'H', 74, ...
           'CO2', 850, 'Lux', 128, 'Sound', 45, 'Soil', 69, ...
           'status', 'Thừa sáng, dư ẩm', ...
           'group', 'Dư ẩm', ...
           'recommendation', 'Giảm ẩm, nguy cơ bệnh nấm/mốc', ...
           'probs', [0.02, 0.01, 99.92, 0.00, 0.00, 0.00, 0.02, 0.01, 0.01, 0.00, 0.00, 0.01]), ...
    
    struct('name', 'Fungal Disease Risk', ...
           'label', 'Not A - 1', ...
           'R', 80, 'EC', 21, 'T', 35, 'H', 68, ...
           'CO2', 1480, 'Lux', 2, 'Sound', 45, 'Soil', 55, ...
           'status', 'Nấm bệnh + thiếu O2, thiếu sáng, thiếu nước', ...
           'group', 'Bệnh nấm', ...
           'recommendation', 'Điều trị nấm, cải thiện thông gió/O2', ...
           'probs', [0.50, 0.10, 0.05, 0.05, 0.10, 0.10, 0.05, 0.02, 0.01, 95.00, 2.00, 2.02]), ...
    
    struct('name', 'Waterlogging + O2 Deficiency', ...
           'label', 'Not A - 2', ...
           'R', 78, 'EC', 19, 'T', 27, 'H', 80, ...
           'CO2', 3300, 'Lux', 1, 'Sound', 45, 'Soil', 74, ...
           'status', 'Thiếu O2, thiếu sáng, thừa nước trầm trọng', ...
           'group', 'Ngập úng/O2', ...
           'recommendation', 'Giảm tưới, tăng O2, cải thiện thoát nước', ...
           'probs', [0.10, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.02, 0.01, 0.50, 98.00, 1.02])
};

%% Class names for all 12 outputs
class_names = {'Chuẩn 1', 'N.Chuẩn 1-1', 'N.Chuẩn 1-2', 'Chuẩn 2', ...
               'N.Chuẩn 2-1', 'N.Chuẩn 2-2', 'Chuẩn 3', 'N.Chuẩn 3-1', ...
               'N.Chuẩn 3-2', 'Not A-1', 'Not A-2', 'Not A-3'};

%% Feature names
feature_names = {'R (kΩ)', 'EC (μS)', 'T (°C)', 'H (%)', ...
                 'CO2 (ppm)', 'Lux', 'Sound (dB)', 'Soil (%)'};

%% Create output folder
output_folder = 'model_output_structured';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% Generate visualization for each scenario
for s = 1:length(scenarios)
    scenario = scenarios{s};
    
    % Create figure with multiple subplots
    fig = figure('Position', [100, 100, 1600, 1000], 'Color', 'w');
    
    % Title
    sgtitle(sprintf('Scenario %d: %s → %s', s, scenario.name, scenario.label), ...
            'FontSize', 18, 'FontWeight', 'bold', 'FontName', 'Arial');
    
    %% Subplot 1: Input Features Radar Chart
    subplot(2, 3, 1);
    
    % Normalize features for radar chart (0-100 scale)
    feat_values = [scenario.R/3, scenario.EC/2, scenario.T*3, scenario.H, ...
                   scenario.CO2/50, scenario.Lux/2, scenario.Sound*2, scenario.Soil];
    feat_values = min(feat_values, 100); % Cap at 100
    
    % Create radar chart
    theta = linspace(0, 2*pi, 9);
    feat_values_plot = [feat_values, feat_values(1)];
    
    polarplot(theta, feat_values_plot, 'b-', 'LineWidth', 2);
    hold on;
    polarplot(theta, ones(1,9)*50, 'k--', 'LineWidth', 1); % Reference circle
    hold off;
    
    ax = gca;
    ax.ThetaTickLabel = [feature_names, feature_names(1)];
    ax.FontSize = 9;
    ax.FontName = 'Arial';
    title('Input Features (Normalized)', 'FontSize', 12, 'FontWeight', 'bold');
    
    %% Subplot 2: Raw Input Values Bar Chart
    subplot(2, 3, 2);
    
    raw_values = [scenario.R, scenario.EC, scenario.T, scenario.H, ...
                  scenario.CO2, scenario.Lux, scenario.Sound, scenario.Soil];
    
    % Normalize to similar scales for visualization
    display_values = [scenario.R/10, scenario.EC, scenario.T, scenario.H, ...
                      scenario.CO2/50, scenario.Lux, scenario.Sound, scenario.Soil];
    
    b = bar(display_values, 'FaceColor', [0.2, 0.6, 0.8]);
    
    % Color code based on severity
    colors = [0.3, 0.7, 0.9; 0.3, 0.7, 0.9; 0.3, 0.7, 0.9; 0.3, 0.7, 0.9;
              0.3, 0.7, 0.9; 0.3, 0.7, 0.9; 0.3, 0.7, 0.9; 0.3, 0.7, 0.9];
    
    % Highlight prominent features
    if scenario.H > 80 || scenario.H < 65
        colors(4, :) = [0.9, 0.3, 0.3]; % Red for H
    end
    if scenario.CO2 > 1000 || scenario.CO2 < 400
        colors(5, :) = [0.9, 0.5, 0.1]; % Orange for CO2
    end
    if scenario.T > 32 || scenario.T < 25
        colors(3, :) = [0.9, 0.5, 0.1]; % Orange for T
    end
    if scenario.Lux > 120 || scenario.Lux < 5
        colors(6, :) = [0.9, 0.5, 0.1]; % Orange for Lux
    end
    if scenario.Soil > 70 || scenario.Soil < 60
        colors(8, :) = [0.9, 0.3, 0.3]; % Red for Soil
    end
    
    % Apply colors to each bar
    b.FaceColor = 'flat';
    b.CData = colors;
    
    set(gca, 'XTick', 1:8, 'XTickLabel', feature_names, 'XTickLabelRotation', 45);
    ylabel('Scaled Value', 'FontSize', 11, 'FontName', 'Arial');
    title('Input Feature Values (Scaled)', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    ax = gca;
    ax.FontSize = 10;
    ax.FontName = 'Arial';
    
    %% Subplot 3: Model Output Probabilities
    subplot(2, 3, [3, 6]);
    
    probs = scenario.probs;
    
    % Sort probabilities for better visualization
    [sorted_probs, sort_idx] = sort(probs, 'descend');
    sorted_names = class_names(sort_idx);
    
    % Color bars based on probability
    colors_probs = zeros(12, 3);
    for i = 1:12
        if sorted_probs(i) > 50
            colors_probs(i, :) = [0.2, 0.7, 0.3]; % Green for high confidence
        elseif sorted_probs(i) > 10
            colors_probs(i, :) = [0.9, 0.7, 0.1]; % Yellow for medium
        else
            colors_probs(i, :) = [0.7, 0.7, 0.7]; % Gray for low
        end
    end
    
    b = barh(sorted_probs, 'FaceColor', 'flat');
    b.CData = colors_probs;
    
    set(gca, 'YTick', 1:12, 'YTickLabel', sorted_names);
    xlabel('Probability (%)', 'FontSize', 11, 'FontName', 'Arial');
    title('Model Output - Class Probabilities', 'FontSize', 12, 'FontWeight', 'bold');
    xlim([0, 105]);
    grid on;
    
    % Add percentage labels
    for i = 1:12
        if sorted_probs(i) > 2
            text(sorted_probs(i) + 2, i, sprintf('%.2f%%', sorted_probs(i)), ...
                 'FontSize', 9, 'FontName', 'Arial', 'VerticalAlignment', 'middle');
        end
    end
    
    ax = gca;
    ax.FontSize = 10;
    ax.FontName = 'Arial';
    
    %% Subplot 4: Health Status Panel
    subplot(2, 3, 4);
    axis off;
    
    % Health status color
    if contains(scenario.group, 'Khỏe')
        status_color = [0.2, 0.8, 0.3];
        status_icon = '✓';
    elseif contains(scenario.group, 'Stress') || contains(scenario.group, 'Thiếu') || contains(scenario.group, 'Dư')
        status_color = [0.9, 0.7, 0.1];
        status_icon = '⚠';
    else
        status_color = [0.9, 0.3, 0.3];
        status_icon = '✗';
    end
    
    % Draw status box
    rectangle('Position', [0.1, 0.5, 0.8, 0.4], 'FaceColor', status_color, ...
              'EdgeColor', 'k', 'LineWidth', 2, 'Curvature', 0.1);
    
    text(0.5, 0.8, 'HEALTH STATUS', 'FontSize', 14, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'Arial');
    text(0.5, 0.65, scenario.status, 'FontSize', 11, ...
         'HorizontalAlignment', 'center', 'FontName', 'Arial');
    
    % Group classification
    text(0.5, 0.3, sprintf('Group: %s', scenario.group), 'FontSize', 12, ...
         'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'FontName', 'Arial');
    
    xlim([0, 1]);
    ylim([0, 1]);
    
    %% Subplot 5: Recommendation Panel
    subplot(2, 3, 5);
    axis off;
    
    % Draw recommendation box
    rectangle('Position', [0.05, 0.2, 0.9, 0.7], 'FaceColor', [0.95, 0.95, 1], ...
              'EdgeColor', [0.2, 0.4, 0.8], 'LineWidth', 2, 'Curvature', 0.1);
    
    text(0.5, 0.85, 'RECOMMENDATION', 'FontSize', 14, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'Arial', 'Color', [0.2, 0.4, 0.8]);
    
    % Wrap recommendation text
    rec_words = strsplit(scenario.recommendation, ' ');
    rec_lines = {};
    current_line = '';
    for w = 1:length(rec_words)
        if length(current_line) + length(rec_words{w}) < 35
            current_line = [current_line, ' ', rec_words{w}];
        else
            rec_lines{end+1} = strtrim(current_line);
            current_line = rec_words{w};
        end
    end
    rec_lines{end+1} = strtrim(current_line);
    
    y_pos = 0.65;
    for i = 1:length(rec_lines)
        text(0.5, y_pos, rec_lines{i}, 'FontSize', 11, ...
             'HorizontalAlignment', 'center', 'FontName', 'Arial');
        y_pos = y_pos - 0.15;
    end
    
    xlim([0, 1]);
    ylim([0, 1]);
    
    %% Save figure
    filename = sprintf('%s/scenario_%d_%s.png', output_folder, s, ...
                      strrep(scenario.name, ' ', '_'));
    print(fig, filename, '-dpng', '-r300');
    
    fprintf('Created visualization: %s\n', filename);
    
    close(fig);
end

fprintf('\n✓ All visualizations created in folder: %s\n', output_folder);
