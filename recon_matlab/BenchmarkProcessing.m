%% Batch Processing for Benchmarks
% BIRI, CMRxRecon2025
% by Yi Zheng

clc; clear; close all;
%% User settings for base paths
% basePath = 'your base file path' ;  
basePath = 'F:\Challenge\ChallengeData\ChallengeData\ChallengeData' ; 
% mainSavePath = 'your address for saving results';       % Folder where results will be saved (modify as needed)
mainSavePath = 'F:\Challenge\ChallengeData\ChallengeBenchmark';

%% List of imaging types
dataTypeList = {'BlackBlood','Cine','Flow2d','LGE','Mapping','Perfusion','T1rho','T1w','T2w'};

%% Parameter meaning
% sampleStatusType = 0 means full kspace data
% sampleStatusType = 1 means subsampled data

% reconType = 0: perform zero-filling recon
% reconType = 1: perform GRAPPA recon
% reconType = 2: perform SENSE recon
% reconType = 3: perform both GRAPPA and SENSE recon

% imgShow = 0: ignore image imshow
% imgShow = 1: image imshow

%% Parameter setting
sampleStatusType = 0; 
reconType = 0;
imgShow = 0;
count = 0;

%% Loop through each imaging type
for i = 1:length(dataTypeList)
    imgType = dataTypeList{i};
    % Construct the FullSample path for the current imaging type
    typePath = fullfile(basePath, 'MultiCoil', imgType, 'TrainingSet', 'FullSample');
    
    % Check if the path exists, if not, skip this type
    if ~exist(typePath, 'dir')
        warning('Path does not exist: %s', typePath);
        continue;
    end
    
    % Get all Center directories
    centerDirs = dir(typePath);
    
    %% Loop through each Center folder
    for j = 1:length(centerDirs)
        centerName = centerDirs(j).name;
        % Ignore hidden folders
        if strcmp(centerName, '.') || strcmp(centerName, '..') || strcmp(centerName, '.DS_Store')
            continue;
        end
        centerPath = fullfile(typePath, centerName);
        
        % Get all device directories in the current Center folder
        deviceDirs = dir(centerPath);
        
        %% Loop through each device folder
        for k = 1:length(deviceDirs)
            deviceName = deviceDirs(k).name;
            % Ignore hidden folders
            if strcmp(deviceName, '.') || strcmp(deviceName, '..') || strcmp(deviceName, '.DS_Store')
                continue;
            end
            % Convert device names: Siemens -> S, UIH -> U
            if contains(deviceName, 'Siemens', 'IgnoreCase', true)
                deviceAbbr = 'S'; bar = [0,0.001];
            elseif contains(deviceName, 'UIH', 'IgnoreCase', true)
                deviceAbbr = 'U'; bar = [];
            else
                deviceAbbr = deviceName;  % For other devices, use original name (modify if needed)
            end
            devicePath = fullfile(centerPath, deviceName);
            
            % Get all patient directories whose name starts with 'P'
            patientDirs = dir(devicePath);
            
            %% Loop through each patient folder
            for m = 1:length(patientDirs)
                patientID = patientDirs(m).name;
                % Ignore hidden folders and only process folders starting with 'P'
                if strcmp(patientID, '.') || strcmp(patientID, '..') || strcmp(patientID, '.DS_Store') || ~startsWith(patientID, 'P')
                    continue;
                end
                patientPath = fullfile(devicePath, patientID);
                
                % Find all .mat files in the patient folder
                matFiles = dir(fullfile(patientPath, '*.mat'));
                
                %% Process each .mat file
                for n = 1:length(matFiles)
                    fileName = matFiles(n).name;
                    % Ignore hidden files if necessary
                    if strcmp(fileName, '.') || strcmp(fileName, '..') || strcmp(fileName, '.DS_Store')
                        continue;
                    end
                    fullKsFilePath = fullfile(patientPath, fileName);
                    % Load the .mat file
                    disp(['Progress start for ', imgType, ', ', centerName, ', ', deviceName, ', '...
                        patientID, ', data ', fileName])
                    fileName = fileName (1:end-4);
                    kspaceData = load(fullKsFilePath);
                    fields = fieldnames(kspaceData);
                    newName = 'kspace';
                    eval([newName ' = kspaceData.' fields{1} ';']);


                    % to reduce the computing burden and space, we only evaluate the central 2 slices
                    % For mapping: we need all weighting for ranking!
                    % For blackblood, it does not have the fifth dimention!
                    % For cine, flow2d and others: use the first 3 time frames for ranking!
                    
                    isBlackBlood = 0;
                    if contains(fullKsFilePath, 'blackblood')
                        [sx,sy,scc,sz] = size(kspace);
                        t = 1;
                        isBlackBlood = 1;
                    else
                        [sx,sy,scc,sz,t] = size(kspace);
                    end

                    isMapping =  contains(fullKsFilePath, 'T1map') || contains(fullKsFilePath, 'T2map');
                    isRho = contains(fullKsFilePath, 'Rho','IgnoreCase', true);
                    isLGE = contains(fullKsFilePath, 'LGE','IgnoreCase', true);

                    if sz < 3
                        sliceToUse = 1:sz;
                    else
                        sliceToUse = (round(sz/2) - 1):(round(sz/2));
                    end
                    
                    if isRho || isLGE
                        sliceToUse = 1:sz;
                    end

                    if isBlackBlood
                        timeFrameToUse = 1;
                    elseif isMapping
                        timeFrameToUse = 1:t;
                    else
                        if t < 3
                            timeFrameToUse = 1:t;
                        else
                            timeFrameToUse = 1:3;
                        end
                    end

                    selectedKspace = kspace(:, :, :, sliceToUse, timeFrameToUse);
                    reconImg = ChallengeRecon(selectedKspace, sampleStatusType, reconType, 0);
                    sampleImg = mat2gray(abs(reconImg(:,:,1,1)));
                    if imgShow ==1
                        figure,imshow(sampleImg);
                    end
                    
                    if length(timeFrameToUse) > 1
                        img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),length(sliceToUse),length(timeFrameToUse)]));
                    else
                        img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),length(sliceToUse)]));
                    end
                    

                    % Construct the new file name: Center_Device_PatientID_Filename.mat
                    newFileName = [centerName, '_', deviceName, '_', patientID, '_', fileName];
                    reconImgFileName = [newFileName, '_recon.mat'];
                    img4rankingFileName = [newFileName, '_ranking.mat'];
                    sampleImgFileName = [newFileName, '.png'];
                    % Construct the save folder path, keeping the same directory structure
                    saveFolder = fullfile(mainSavePath, 'Benchmark', imgType, centerName, deviceName, patientID);
                    if ~exist(saveFolder, 'dir')
                        mkdir(saveFolder);
                    end
                    
                    % Save the reconstruction result into the specified file
                    save(fullfile(saveFolder, reconImgFileName), 'reconImg');
                    save(fullfile(saveFolder, img4rankingFileName), 'img4ranking');
                    imwrite(sampleImg, fullfile(saveFolder, sampleImgFileName));

                end
            end
        end
    end
end

disp('All processing complete!');