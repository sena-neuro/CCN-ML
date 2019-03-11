
function [] = convert_data_to_libsvm_format

% ClassificationStyle - all time points for the video period are used, not time-resolved

load('video_robot.mat'); % NoOfChannels x TimePoints x NoOfTrials matrix
load('video_android.mat'); % NoOfChannels x TimePoints x NoOfTrials matrix
load('video_human.mat'); % NoOfChannels x TimePoints x NoOfTrials matrix


%% Prepare data for classification by using the whole time series - not time-resolved

for i = 1:size(video_robot,3) % ROBOT
    temp = video_robot(:,101:400,i); % take only the video period (first 600 ms), not the baseline (which is 1-100 indices)
    temp2 = temp(:);
    trials_robot(i,:) = temp2; % NoOfTrials x AllChannelsAndTimePoints_concatanated
end

for i = 1:size(video_android,3) % ANDROID
    temp = video_android(:,101:400,i); % take only the video period (first 600 ms), not the baseline (which is 1-100 indices)
    temp2 = temp(:);
    trials_android(i,:) = temp2; % NoOfTrials x AllChannelsAndTimePoints_concatanated
end

for i = 1:size(video_human,3) % HUMAN
    temp = video_human(:,101:400,i); % take only the video period (first 600 ms), not the baseline (which is 1-100 indices)
    temp2 = temp(:);
    trials_human(i,:) = temp2; % NoOfTrials x AllChannelsAndTimePoints_concatanated
end


%% Divide data into training and test sets

% ROBOT
order1 = randperm(size(trials_robot,1));
robot_shuff = trials_robot(order1,:);

cond1_noOfTrials = size(trials_robot,1);
trainset1_noOftrials = round(cond1_noOfTrials*0.8); % 80% data for training
testset1_noOftrials = cond1_noOfTrials - trainset1_noOftrials; % 20% data for test
trainset1 = 1:trainset1_noOftrials;
testset1 = (trainset1_noOftrials+1):cond1_noOfTrials;

robot_training = robot_shuff(trainset1,:);
robot_test = robot_shuff(testset1,:);


% ANDROID
order2 = randperm(size(trials_android,1));
android_shuff = trials_android(order2,:);

cond2_noOfTrials = size(trials_android,1);
trainset2_noOftrials = round(cond2_noOfTrials*0.8);
testset2_noOftrials = cond2_noOfTrials - trainset2_noOftrials;
trainset2 = 1:trainset2_noOftrials;
testset2 = (trainset2_noOftrials+1):cond2_noOfTrials;

android_training = android_shuff(trainset2,:);
android_test = android_shuff(testset2,:);


% HUMAN
order3 = randperm(size(trials_human,1));
human_shuff = trials_human(order3,:);

cond3_noOfTrials = size(trials_human,1);
trainset3_noOftrials = round(cond3_noOfTrials*0.8);
testset3_noOftrials = cond3_noOfTrials - trainset3_noOftrials;
trainset3 = 1:trainset3_noOftrials;
testset3 = (trainset3_noOftrials+1):cond3_noOfTrials;

human_training = human_shuff(trainset3,:);
human_test = human_shuff(testset3,:);


%%  Write the training data into a file in LIBSVM format
fileID = fopen('data_training', 'a');
colon = ':';
for i = 1:size(robot_training,1)
    fprintf(fileID, '%d\t', 0); % Assign class label 0 to robot
    for j = 1:size(robot_training,2)
        fprintf(fileID, '%d%s%d\t', j, colon, robot_training(i,j));
    end
    fprintf(fileID, '\n');
end

for i = 1:size(android_training,1)
    fprintf(fileID, '%d\t', 1); % Assign class label 1 to android
    for j = 1:size(android_training,2)
        fprintf(fileID, '%d%s%d\t', j, colon, android_training(i,j));
    end
    fprintf(fileID, '\n');
end

for i = 1:size(human_training,1)
    fprintf(fileID, '%d\t', 2); % Assign class label 2 to human
    for j = 1:size(human_training,2)
        fprintf(fileID, '%d%s%d\t', j, colon, human_training(i,j));
    end
    fprintf(fileID, '\n');
end

for i = 1:size(scr_training,1)
    fprintf(fileID, '%d\t', 3);
    for j = 1:size(scr_training,2)
        fprintf(fileID, '%d%s%d\t', j, colon, scr_training(i,j));
    end
    fprintf(fileID, '\n');
end
fclose(fileID);


%%  Write the test data into a file in LIBSVM format
fileID = fopen('data_test.t', 'a');
colon = ':';
for i = 1:size(robot_test,1)
    fprintf(fileID, '%d\t', 0);
    for j = 1:size(robot_test,2)
        fprintf(fileID, '%d%s%d\t', j, colon, robot_test(i,j));
    end
    fprintf(fileID, '\n');
end

for i = 1:size(android_test,1)
    fprintf(fileID, '%d\t', 1);
    for j = 1:size(android_test,2)
        fprintf(fileID, '%d%s%d\t', j, colon, android_test(i,j));
    end
    fprintf(fileID, '\n');
end

for i = 1:size(human_test,1)
    fprintf(fileID, '%d\t', 2);
    for j = 1:size(human_test,2)
        fprintf(fileID, '%d%s%d\t', j, colon, human_test(i,j));
    end
    fprintf(fileID, '\n');
end

end
