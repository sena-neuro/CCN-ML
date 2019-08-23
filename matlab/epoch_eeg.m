clc;
clear all;
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
%%%pth='/auto/data2/oelmas/EEG_AgentPerception_NAIVE/Data/';
pth='/home/sena/Desktop/CCN Lab/EEG_AgentPerception_NAIVE/Data/Still/';

folders = dir(pth);
robot   = { 'S 51'  'S 52'  'S 53'  'S 54'  'S 55'  'S 56'  'S 57'  'S 58'};
android = { 'S 61' 'S 62' 'S 63' 'S 64' 'S 65' 'S 66' 'S 67' 'S 68'};
human   = { 'S 71' 'S 72' 'S 73' 'S 74' 'S 75' 'S 76' 'S 77' 'S 78'};
% robot   = {'S101' 'S102' 'S103' 'S104' 'S105' 'S106' 'S107' 'S108'};
% android = {'S111' 'S112' 'S113'  'S114' 'S115' 'S116'  'S117' 'S118'};
% human   = {'S121' 'S122' 'S123'  'S124' 'S125' 'S126'  'S127' 'S128'};
channels = {'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5' ,'FC1', 'C3', 'T7', 'CP5', 'CP1'...
    'Pz' ,'P3' ,'P7' ,'O1' ,'Oz' ,'O2' ,'P4', 'P8' ,'CP6' ,'CP2' ,'Cz' ,'C4' ,'T8' ...
    'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7'...
    'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4'...
    'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2'... 
    'AF4' ,'AF8'};
list = containers.Map;
list('robot') = robot;
list('android') = android;
list('human') = human;


for k=4:4%length(folders)
    if(folders(k).isdir && ~strcmp(folders(k).name,'.') && ~strcmp(folders(k).name,'..') )
       folder_name = folders(k).name;
       file_path = strcat(pth,folder_name,'/');
       files = dir(file_path);
       file_name = files(3).name;
       fprintf('File Name: %s  File Path: %s\n',file_name,file_path);
       for i=1:list.Count
           EEG = pop_loadset('filename',file_name,'filepath',file_path);
           [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
           EEG = eeg_checkset( EEG );
           types = keys(list);
           t = char(types(i));
           spth = strcat(pth, folder_name, '/');
           name = strcat('subject', int2str(k+1)); % change to k-1 when starting from subj2; k+1 for starting from subj4
           
           % Epoching
           EEG = pop_epoch( EEG, list(t), [-0.2  0.600], 'newname', strcat(name,'_epochs_',t), 'epochinfo', 'yes');
           [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'savenew',strcat(spth, 'steps/', name, '_', t, '_step1_epchs.set'),'overwrite','on','gui','off'); 
           EEG = eeg_checkset( EEG );
           
           % Baseline removal
           EEG = pop_rmbase( EEG, [-200 0]);
           [ALLEEG, EEG, ~] = pop_newset(ALLEEG, EEG, 2, 'savenew', strcat(spth, 'steps/', name, '_', t, '_step2_rem_bas.set'),'gui','off'); 
           EEG = eeg_checkset( EEG );
           
           % Select channels
           EEG = pop_select( EEG,'channel', channels);
           [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'savenew',strcat(spth, 'steps/', name, '_', t, '_step3_chnl','.set'),'overwrite','on','gui','off'); 
           eeg_data = EEG.data;
           save(strcat(spth, 'subject_',int2str(k+1),'_',t,'.mat'), 'eeg_data'); % change for subj2
       end
    end
end
