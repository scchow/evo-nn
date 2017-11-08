function plot_nonadaptive()
close all 

setPlot();
epochs = 3000;
nights = 10;
capacity = 10;
numTrials = 3;
numAgents = 100;

date = '2017-11-03_10-02-19';

% numDisabled = {'0', '20', '50', '60', '70', '90'};
numDisabled = {'0'};

% paths = arrayfun(@(x) strcat('../build/Results/', date, '/MultiNightBarQ/non-adaptive', ...
%     "/", num2str(numAgents),'_agents/',x,'_disabled'),numDisabled);

paths = arrayfun(@(x) strcat('results_11-8/final/MultiNightBarQ/non-adaptive', ...
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



markers = ['o'; 'v'; 'x'; 's'; 'p'; '^'; '<'; '>'];
linestyles = {'-'; '--'; '-.'; ':'};

increment = 200;
maxEpoch = 3000;
dict_keys = numDisabled;

for i = 1:length(dict_keys)
    key = dict_keys{i};
    value = dataDict(key);
    x_axis = value(1:increment:maxEpoch,1);
    y_axis = value(1:increment:maxEpoch,2);
    errors = value(1:increment:maxEpoch,3);
    e = errorbar(x_axis, y_axis, errors, ...
        'Marker', markers(mod(i,length(markers))), ...
        'Linestyle', linestyles{1+mod(i, length(linestyles))} ...
        );
    hold on

end

title(strcat('Performance vs Number of Epochs for ', num2str(nights), ...
    ' Nights of ', num2str(capacity), ' Capacity with ', num2str(numAgents), ' Adaptive ', 'Agents'));
xlabel('Number of Epochs');
ylabel('Performance (max 100)');
legend(numDisabled);

end

function setPlot()

width = 3;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

% The properties we've been using in the figures
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz

% Set the default Size for display
defpos = get(0,'defaultFigurePosition');
set(0,'defaultFigurePosition', [defpos(1) defpos(2) width*100, height*100]);

% Set the defaults for saving/printing to a file
set(0,'defaultFigureInvertHardcopy','on'); % This is the default anyway
set(0,'defaultFigurePaperUnits','inches'); % This is the default anyway
defsize = get(gcf, 'PaperSize');
left = (defsize(1)- width)/2;
bottom = (defsize(2)- height)/2;
defsize = [left, bottom, width, height];
set(0, 'defaultFigurePaperPosition', defsize);
end