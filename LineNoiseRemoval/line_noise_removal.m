%% Example: Running the pipeline on a directory of EEG files

%% Set up the input and the output directories
dataFolder = 'F:BACKUP F\edf\NMTDataset';
outdirtemp = 'E:\TempDataNMT';



trainAndTestFiles = dir(fullfile(dataFolder,'**','*.edf'));
trainAndTestFiles = fullfile({trainAndTestFiles.folder},{trainAndTestFiles.name});
%% Make the output directory if needed
if ~exist(outdirtemp, 'dir')
    mkdir(outdirtemp)
end

%% Set up the params structure
params = struct();

params.lineFrequencies = [60, 120, 180, 240];
params.detrendChannels = 1:21;
params.detrendType = 'high pass';
params.detrendCutoff = 1;
params.referenceType = 'robust';
params.meanEstimateType = 'median';
params.interpolationOrder = 'post-reference';
params.removeInterpolatedChannels = true;
params.keepFiltered = false;


%% Get the filelist
fileList = trainAndTestFiles
%% Run the pipeline
for k = 1:length(fileList)
    [thisfolder, thisName, ~] = fileparts(fileList{k});
    if contains(thisfolder,"\train")
        if contains(thisfolder,"\normal")
            outdir = append(outdirtemp,'\normal\train');
            
        elseif contains(thisfolder,"\abnormal")
            outdir = append(outdirtemp,'\abnormal\train');
        end
    elseif contains(thisfolder,"\eval")
        if contains(thisfolder,"\normal")
           outdir = append(outdirtemp,'\normal\eval');
       elseif contains(thisfolder,"\abnormal")
            outdir = append(outdirtemp,'\abnormal\eval');
        end
    end
    fprintf("test")
    EEG = pop_biosig(fileList{k});
    
    params.name = thisName;
    [EEG, params, computationTimes] = justcleannoise(EEG, params);
    fprintf('Computation times (seconds):\n   %s\n', ...
        getStructureString(computationTimes));
   
    fname = [outdir filesep thisName '.set'];
    save(fname, 'EEG', '-mat', '-v7.3'); 
    if strcmpi(params.errorMsgs, 'verbose')
        outputPrepParams(params, 'Prep parameters (non-defaults)');
        outputPrepErrors(EEG.etc.noiseDetection, 'Prep error status');
     end
        
end
