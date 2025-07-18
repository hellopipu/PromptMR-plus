%% Recon code for CMRxRecon25 
% Test only for benchmarks
clc; close all; clear
%% set info
coilInfo = 'MultiCoil\';  % singleCoil is not avalaible for PI recon
setName = 'TrainingSet\'; % options: 'TrainingSet/', ValidationSet/', 'TestSet/'
% taskType = {'Task1', 'Task2'};
% dataTypeList = {'Aorta', 'BlackBlood', 'Cine', 'Flow2d',   'Mapping', 'Tagging'};
% input and output folder paths
basePath = 'F:\Challenge\ChallengeData\test\ChallengeData';
mainSavePath = 'F:\Challenge\ChallengeData\test\result';

% Open a file selection dialog and filter to show only .mat files
[file, path] = uigetfile('*.mat', 'Please select a MAT file');

% Check if a file was selected
if isequal(file, 0)
    error('No file selected. Please run the program again and select a MAT file.');
else
    fullFilePath = fullfile(path, file);
    disp(['File selected: ', fullFilePath]);
    % Load the .mat file
    kspaceData = load(fullFilePath);
end

%% parameter meaning
% sampleStatusType = 0 means full kspace data
% sampleStatusType = 1 means subsampled data

% reconType = 0: perform zero-filling recon
% reconType = 1: perform GRAPPA recon
% reconType = 2: perform SENSE recon
% reconType = 3: perform both GRAPPA and SENSE recon

% imgShow = 0: ignore image imshow
% imgShow = 1: image imshow

%%
sampleStatusType = 0; 
reconType = 0;
imgShow = 0;

%%
disp('Progress start for subject')
fields = fieldnames(kspaceData);
newName = 'kspace';
eval([newName ' = kspaceData.' fields{1} ';']);

% to reduce the computing burden and space, we only evaluate the central 2 slices
% For cine, flow2d, aorta, tagging: use the first 3 time frames for ranking!
% For mapping: we need all weighting for ranking!
% For blackblood, it does not have the fifth dimention!
isBlackBlood = 0;
if contains(fullFilePath, 'blackblood')
    [sx,sy,scc,sz] = size(kspace);
    t = 1;
    isBlackBlood = 1;
else
    [sx,sy,scc,sz,t] = size(kspace);
end
isMapping =  contains(fullFilePath, 'T1map') || contains(fullFilePath, 'T2map.mat');
isRho = contains(fullFilePath, 'Rho','IgnoreCase', true);
isLGE = contains(fullFilePath, 'LGE','IgnoreCase', true);
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


if sampleStatusType == 1
    %  running the code on the training set requires iterating over the undersampled template
    if contains(setName, 'Train')
        maskFiles = dir(fullfile(fullMaskDirPath, '*.mat'));
        for iMaskPaths = 1:length(maskFiles)
            maskedKspace = zeros(sx, sy, scc, sz, t);
            isRadial = 0;
            maskFileInfo = maskFiles(iMaskPaths);
            if isempty(strfind(maskFileInfo.name, replace(ksFileInfo.name, '.mat', '')))
                continue;
            end
            fullMaskFilePath = fullfile(fullMaskDirPath, maskFileInfo.name);
            if ~isempty(strfind(maskFileInfo.name, 'Radial'))
                isRadial = 1;
            end
            load(fullMaskFilePath);  % 'mask' is the key of us mask
            if length(size(mask))>2
                for iFrame = 1:size(mask,3)
                    maskedKspace(:,:,:,:,iFrame) = kspace(:,:,:,:,iFrame) .* mask(:,:,iFrame);
                end
            else
                maskedKspace = kspace .* mask;
            end
            selectedKspace = maskedKspace(:, : ,:, sliceToUse, timeFrameToUse);


            % recon
            reconImg = ChallengeRecon(selectedKspace, sampleStatusType, reconType, imgShow, isRadial);
            if length(timeFrameToUse) > 1
                img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),length(sliceToUse),length(timeFrameToUse)]));
            else
                img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),length(sliceToUse)]));
            end
        end

        % the data in the validationset and testset are undersampled
    else
        isRadial = 0;
        if ~isempty(strfind(ksFileInfo.name, 'Radial'))
            isRadial = 1;
        end
        try
            selectedKspace = kspace(:, :, :, sliceToUse, timeFrameToUse);
        catch
            disp
        end
        reconImg = ChallengeRecon(selectedKspace, sampleStatusType, reconType, imgShow, isRadial);
        if length(timeFrameToUse) > 1
            img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),length(sliceToUse),length(timeFrameToUse)]));
        else
            img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),length(sliceToUse)]));
        end
    end

    % mkdir and save
    fullSaveDirPath = fullfile(mainSavePath, coilInfo, dataType, setName, taskType, ksFileDirInfo.name);
    if contains(setName, 'Train')
        fullSavePath = fullfile(fullSaveDirPath, replace(maskFileInfo.name, '_mask', ''));
    else
        fullSavePath = fullfile(fullSaveDirPath, ksFileInfo.name);
    end

    % create any missing folders in the save path
    if ~exist(fullSaveDirPath, 'dir')
        createRecursiveDir(fullSaveDirPath)
    end

    save(fullSavePath, 'img4ranking');

    %%
else
    selectedKspace = kspace(:, :, :, sliceToUse, timeFrameToUse);
    reconImg = ChallengeRecon(selectedKspace, sampleStatusType, reconType, imgShow);
    isSiemens = contains(fullFilePath, 'Siemens','IgnoreCase', true);
    if isSiemens
         figure,imshow(abs(reconImg(:,:,1,1)),[0,0.001]); 
    else
         figure,imshow(abs(reconImg(:,:,1,1)),[]);
    end
       

    if length(timeFrameToUse) > 1
        img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),length(sliceToUse),length(timeFrameToUse)]));
    else
        img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),length(sliceToUse)]));
    end
    fullSaveDirPath = fullfile(mainSavePath, coilInfo, dataType, setName, taskType, 'FullSample', ksFileDirInfo.name);
    fullSavePath = fullfile(fullSaveDirPath, ksFileInfo.name);

    if ~exist(fullSaveDirPath, 'dir')
        createRecursiveDir(fullSaveDirPath)
    end
    save(fullSavePath, 'img4ranking');
end