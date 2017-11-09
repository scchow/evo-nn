function plot_numLearning(varargin)
varargin
numAgents = 100;
legendLoc = 'SouthEast';
if length(varargin)==1
    numAgents = varargin;
elseif length(varargin)==2
    numAgents = varargin{1};
    legendLoc = varargin{2};
end
figure;
epochs = 3000;
nights = 10;
capacity = 10;
numTrials = 20;
% numAgents = 200;

temperatures = {'50'; '100'; '300'; '500'; '1000'};
tempLegend = arrayfun(@(x) strcat('$\tau = ', x,'$'), temperatures);

% paths = arrayfun(@(x) strcat('../build/Results/adaptive_softmax/temp_', x, ...
%     "/", num2str(numAgents),"_agents/0_disabled"),temperatures);
% 
% temperatures = vertcat(['All Agents Learning'], temperatures);
% paths = vertcat('../build/Results/2017-11-03_10-02-19/MultiNightBarQ/non-adaptive/100_agents/0_disabled', paths);

% paths = arrayfun(@(x) strcat('results_11-8/final/MultiNightBarQ/adaptive_softmax_G-distributed/temp_', x, ...
%     "/", num2str(numAgents),"_agents/0_disabled"),temperatures);
% 
% temperatures = vertcat({'All Agents Learning'}, temperatures);
% paths = vertcat(strcat('results_11-8/final/MultiNightBarQ/non-adaptive', '/', num2str(numAgents),'_agents/0_disabled'), paths);

% On Desktop
paths = arrayfun(@(x) strcat('../build/Results/final_discount0/MultiNightBarQ/adaptive_softmax_G-distributed/temp_', x, ...
    "/", num2str(numAgents),"_agents/0_disabled"),temperatures);

dataDict = containers.Map();

for i = 1:size(paths)
    temp = temperatures{i};

    path = paths(i);
    
    csvFname = '/numLearning.csv';
    
    trialFolders = arrayfun(@(x) strcat('/trial_',num2str(x)), 0:numTrials-1, 'UniformOutput', false);
    file = strcat(path, '/trial_0', csvFname)
    trial0 = csvread(file);
    data = zeros(size(trial0, 1), numTrials);
    data_numLearning = zeros(size(trial0,1), numTrials);
    
    for j = 1:numTrials
       trialData =  csvread(strcat(path, trialFolders(j), csvFname));
       data(:,j) = trialData(:,2);
    end
    
    meanStd = zeros(size(trial0, 1), 3);
    meanStd(:,1) = trial0(:,1);
    meanStd(:,2) = mean(data, 2);
    meanStd(:,3) = std(data,0, 2)./sqrt(numTrials);
    
    dataDict(temp) = meanStd;
    
end




markers = ['o'; 'v'; 's'; '^'; 'd'; 'p';'x'];
linestyles = {'-.'; '-'; '--'};
colors = get(gca, 'colororder');
% '-' = baseline
% 'o' for original
set(gcf, 'Position', [1000, 800, 560, 420])
set(gca, 'FontName', 'Times New Roman');
lw = 1;
fs = 14;

increment = 200;
maxEpoch = 3000;
dict_keys = temperatures;


plotHandles = zeros(length(dict_keys),1);
errHandles = zeros(length(dict_keys),1);
sampleHandles = zeros(length(dict_keys),1);

for i = 1:length(dict_keys)
    key = dict_keys{i};
    value = dataDict(key);
    epochs = value(:,1);
    means = value(:,2);
    stderr = value(:,3);
    
    x_axis = value(1:increment:maxEpoch,1);
    y_axis = value(1:increment:maxEpoch,2);
    errors = value(1:increment:maxEpoch,3);
%     e = errorbar(x_axis, y_axis, errors, ...
%         'Marker', markers(mod(i,length(markers))), ...
%         'Linestyle', linestyles{1+mod(i, length(linestyles))} ...
%         );
%     hold on

    % Plot line
    
    ls = linestyles{1 + mod(i, length(linestyles))};
    c = colors(i+1,:);
    mkr = markers(mod(i,length(markers)));
    plotHandles(i) = plot(epochs, means, 'LineStyle', ls, 'LineWidth', lw, 'Color', c);
    hold on
    errHandles(i) = errorbar(x_axis, y_axis, errors, ...
        'LineStyle', 'None', 'Marker', mkr ,'MarkerFaceColor', c, 'Color', c);
    sampleHandles(i) = errorbar(x_axis(1), y_axis(1), errors(1), ...
        'LineStyle', ls, 'Marker', mkr,'MarkerFaceColor', c, 'Color', c, 'LineWidth', lw);
    
end

% title(strcat('Performance vs Number of Epochs for ', num2str(nights), ...
%     ' Nights of ', num2str(capacity), ' Capacity with ', num2str(numAgents), ' Adaptive ', 'Agents'));

xlabel('Epoch', 'FontSize', fs, 'Interpreter', 'latex');
ylabel('Number Agents Learning', 'FontSize', fs, 'Interpreter', 'latex');
legend(sampleHandles, tempLegend, 'Location', legendLoc, 'Interpreter', 'latex');
set(gca,'fontname','Times New Roman','FontSize',fs)
grid on 
if numAgents == 100
    ylabel('Number Agents Learning (max 100)', 'FontSize', fs, 'Interpreter', 'latex');
    savefig('bar_numLearning_100agents.fig')
    export_fig(gcf, 'bar_numLearning_100agents.pdf', '-trans');

elseif numAgents == 150
    ylabel('Number Agents Learning (max 150)', 'FontSize', fs, 'Interpreter', 'latex');
    savefig('bar_numLearning_150agents.fig')
    export_fig(gcf, 'bar_numLearning_150agents.pdf', '-trans');

elseif numAgents == 200
    ylabel('Number Agents Learning (max 200)', 'FontSize', fs, 'Interpreter', 'latex');
    savefig('bar_numLearning_200agents.fig')
    export_fig(gcf, 'bar_numLearning_200agents.pdf', '-trans');

else
'invalid number of agents, cant export_fig'
end
end
