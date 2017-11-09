function plot_nonadaptive(varargin)
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

date = '2017-11-03_10-02-19';
if numAgents == 100
    numDisabled = {'0', '20', '50', '70', '90'};

elseif numAgents == 150
    numDisabled = {'0', '20', '50', '80', '100' '120'};

elseif numAgents == 200
    numDisabled = {'0', '50', '80', '100', '150', '170'};

else
    'invalid number of agents'
end
    
% numDisabled = {'0'};

% paths = arrayfun(@(x) strcat('../build/Results/', date, '/MultiNightBarQ/non-adaptive', ...
%     "/", num2str(numAgents),'_agents/',x,'_disabled'),numDisabled);

% paths = arrayfun(@(x) strcat('results_11-8/final_discount0/MultiNightBarQ/non-adaptive', ...
% paths = arrayfun(@(x) strcat('results_11-8/final/MultiNightBarQ/non-adaptive', ...
%     "/", num2str(numAgents),'_agents/',x,'_disabled'),numDisabled);

paths = arrayfun(@(x) strcat('../build/Results/final_discount0/MultiNightBarQ/non-adaptive', ...
    "/", num2str(numAgents),'_agents/',x,'_disabled'),numDisabled);


dataDict = containers.Map();

for i = 1:size(paths,2)
    nDisabled = numDisabled{i};

    path = paths(i)
    
    csvFname = '/results.csv';
    
    trialFolders = arrayfun(@(x) strcat('/trial_',num2str(x)), 0:numTrials-1, 'UniformOutput', false);
    file = strcat(path, '/trial_0', csvFname);
    trial0 = csvread(file);
    data = zeros(size(trial0, 1), numTrials);
    
    for j = 1:numTrials
       trialData =  csvread(strcat(path, trialFolders(j), csvFname));
       data(:,j) = trialData(:,2);
    end
    
    meanStd = zeros(size(trial0, 1), 3);
    meanStd(:,1) = trial0(:,1);
    meanStd(:,2) = mean(data, 2);
    meanStd(:,3) = std(data,0, 2)./sqrt(numTrials);
    
    dataDict(nDisabled) = meanStd;
    
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
maxEpoch = 2000;
dict_keys = numDisabled;

plotHandles = zeros(length(dict_keys),1);
errHandles = zeros(length(dict_keys),1);
sampleHandles = zeros(length(dict_keys),1);

for i = 1:length(dict_keys)
    key = dict_keys{i};
    value = dataDict(key);
    epochs = value(1:maxEpoch,1);
    means = value(1:maxEpoch,2);
    stderr = value(1:maxEpoch,3);
    
    x_axis = value(1:increment:maxEpoch,1);
    y_axis = value(1:increment:maxEpoch,2);
    errors = value(1:increment:maxEpoch,3);
%     e = plot(x_axis, y_axis,... errors, ...
%         'Marker', markers(mod(i,length(markers))), ...
%         'Linestyle', linestyles{1+mod(i, length(linestyles))} ...
%         );
    
    % Plot line
    
    ls = linestyles{1 + mod(i, length(linestyles))};
    c = colors(i,:);
    mkr = markers(mod(i,length(markers)));
    plotHandles(i) = plot(epochs, means, 'LineStyle', ls, 'LineWidth', lw, 'Color', c);
    hold on
    errHandles(i) = errorbar(x_axis, y_axis, errors, ...
        'LineStyle', 'None', 'Marker', mkr , 'MarkerFaceColor', c, 'Color', c);
    sampleHandles(i) = errorbar(x_axis(1), y_axis(1), errors(1), ...
        'LineStyle', ls, 'Marker', mkr, 'MarkerFaceColor', c, 'Color', c,'LineWidth', lw);
    

end

% title(strcat('Performance vs Epochs for ', num2str(nights), ...
%     ' Nights of ', num2str(capacity), ' Capacity with ', num2str(numAgents), ' Adaptive ', 'Agents'));

xlabel('Epoch', 'FontSize', 14, 'Interpreter', 'latex');
legend(sampleHandles,numDisabled, 'Location', legendLoc, 'Interpreter', 'latex');
ylim([10,100]);

set(gca,'fontname','Times New Roman','FontSize',fs)
grid on 

if numAgents == 100
    ylabel('Performance (max 100)', 'FontSize', fs, 'Interpreter', 'latex');
    savefig('bar_nonadaptive_100agents.fig')
    export_fig(gcf, 'bar_nonadaptive_100agents.pdf', '-trans');

elseif numAgents == 150
    ylabel('Performance (max 90)', 'FontSize', fs, 'Interpreter', 'latex');
    savefig('bar_nonadaptive_150agents.fig')
    export_fig(gcf, 'bar_nonadaptive_150agents.pdf', '-trans');

elseif numAgents == 200
    ylabel('Performance (max 90)', 'FontSize', fs, 'Interpreter', 'latex');
    savefig('bar_nonadaptive_200agents.fig')
    export_fig(gcf, 'bar_nonadaptive_200agents.pdf', '-trans');

else
'invalid number of agents, cant export_fig'

end
end