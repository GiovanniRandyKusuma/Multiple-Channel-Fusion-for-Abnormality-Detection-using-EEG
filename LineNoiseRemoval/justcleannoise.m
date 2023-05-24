function [EEG, params, computationTimes] = justcleannoise(EEG, params)
    computationTimes= struct( 'detrend', 0, 'lineNoise', 0, 'reference', 0);
    errorMessages = struct('status', 'good', 'boundary', 0, ...
                   'detrend', 0, 'lineNoise', 0, 'reference', 0, ...
                   'postProcess', 0);
    [backupOptionsFile, currentOptionsFile, warningsState] = setupForEEGLAB();
    finishup = onCleanup(@() cleanup(backupOptionsFile, currentOptionsFile, ...
        warningsState));
    if isfield(EEG.etc, 'noiseDetection')
        warning('EEG.etc.noiseDetection already exists and will be cleared\n')
    end
    if ~exist('params', 'var')
        params = struct();
    end
    if ~isfield(params, 'name')
        params.name = ['EEG' EEG.filename];
    end
    EEG.etc.noiseDetection = ...
        struct('name', params.name, 'version', getPrepVersion, ...
               'originalChannelLabels', [], ...
               'errors', [], 'boundary', [], 'detrend', [], ...
               'lineNoise', [], 'reference', [], 'postProcess', [], ...
               'interpolatedChannelNumbers', [], 'removedChannelNumbers', [], ...
               'stillNoisyChannelNumbers', [], 'fullReferenceInfo', false);
    EEG.data = double(EEG.data);   % Don't monkey around -- get into double
    EEG.etc.noiseDetection.originalChannelLabels = {EEG.chanlocs.labels};

%% Check for the general defaults
    try
        defaults = getPrepDefaults(EEG, 'general');
        [params, errors] = checkPrepDefaults(params, params, defaults);
        if ~isempty(errors)
            error('prepPipeline:GeneralDefaultError', ['|' sprintf('%s|', errors{:})]);
        end
    catch mex
        warning('[%s] %s: Prep could not begin processing the EEG', ...
           mex.identifier, getReport(mex, 'extended', 'hyperlinks', 'on')); 
        return;
    end

%% Check for boundary events
    fprintf('Checking for boundary events\n');
    try
        defaults = getPrepDefaults(EEG, 'boundary');
        [boundaryOut, errors] = checkPrepDefaults(params, struct(), defaults);
        if ~isempty(errors)
            error('boundary:BadParameters', ['|' sprintf('%s|', errors{:})]);
        end
        EEG.etc.noiseDetection.boundary = boundaryOut;
        if ~boundaryOut.ignoreBoundaryEvents && ...
                isfield(EEG, 'event') && ~isempty(EEG.event)
            eTypes = find(strcmpi({EEG.event.type}, 'boundary'));
            if ~isempty(eTypes)
                error('boundary:UnremovedBoundary', ['Dataset ' params.name  ...
                    ' has boundary events: [' getListString(eTypes) ...
                    '] which are treated as discontinuities unless ' ...
                    'set to ignore. Prep cannot continue']);
            end
        end
    catch mex
        errorMessages.boundary = ['prepPipeline bad boundary events: ' ...
             getReport(mex, 'basic', 'hyperlinks', 'off')];
        errorMessages.status = 'unprocessed';
        EEG.etc.noiseDetection.errors = errorMessages;
        if strcmpi(params.errorMsgs, 'verbose')
            warning('[%s]\n%s', mex.identifier, ...
               getReport(mex, 'extended', 'hyperlinks', 'on')); 
        end
        return;
    end

%% Part II:  HP the signal for detecting bad channels
    fprintf('Preliminary detrend to compute reference\n');
    try
        tic
        [EEGNew, detrend] = removeTrend(EEG, params);
        EEG.etc.noiseDetection.detrend = detrend;
        % Make sure detrend defaults are available for referencing
        defaults = getPrepDefaults(EEG, 'detrend');
        params = checkPrepDefaults(detrend, params, defaults); 
        computationTimes.detrend = toc;
    catch mex
        errorMessages.detrend = ['prepPipeline failed removeTrend: ' ...
             getReport(mex, 'basic', 'hyperlinks', 'off')];
        errorMessages.status = 'unprocessed';
        EEG.etc.noiseDetection.errors = errorMessages;
        if strcmpi(params.errorMsgs, 'verbose')
            warning('[%s]\n%s', mex.identifier, ...
                getReport(mex, 'extended', 'hyperlinks', 'on')); 
        end
        return;
    end
 
%% Part III: Remove line noise
    fprintf('Line noise removal\n');
    try
        tic
        [EEGNew] = removeTrend(EEG, params);
        [EEGClean, lineNoise] = removeLineNoise(EEGNew, params);
        lineChannels = lineNoise.lineNoiseChannels;
        EEG.data(lineChannels, :) = EEG.data(lineChannels, :) ...
             - EEGNew.data(lineChannels, :) + EEGClean.data(lineChannels, :); 
        clear EEGNew;
        computationTimes.lineNoise = toc;
    catch mex
        errorMessages.lineNoise = ['prepPipeline failed removeLineNoise: ' ...
            getReport(mex, 'basic', 'hyperlinks', 'off')];
        errorMessages.status = 'unprocessed';
        EEG.etc.noiseDetection.errors = errorMessages;
        if strcmpi(params.errorMsgs, 'verbose')
            warning('[%s]\n%s', mex.identifier, ...
                getReport(mex, 'extended', 'hyperlinks', 'on')); 
        end
        return;
    end 
end

%% Cleanup callback
function cleanup(backupFile, currentFile, warningsState)
% Restore EEGLAB options file and warning settings 
   restoreEEGOptions(backupFile, currentFile);
   warning(warningsState);
end % cleanup