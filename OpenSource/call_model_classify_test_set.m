clc; clear
% close all
%%
for user = 1 : 2
    switch user
        case 1
            test_object = 'A';
        case 2
            test_object = 'B';
    end
    result_save_path = ['results/',test_object,'testset/'];
    if exist(result_save_path,'dir') == 0
        mkdir(result_save_path)
    end
for feat_chose = 1 : 10
    switch feat_chose
        case 1
            feature = 'RDTI'; 
        case 2
            feature = 'DTI';
        case 3
            feature = 'RTI';
        case 4
            feature = 'HATI';
        case 5
            feature = 'VATI';
        case 6
            feature = 'HVTI';
        case 7
            feature = 'DT_HAT_VAT_3Channel';
        case 8
            feature = 'DT_RT_HVT_3Channel';
        case 9
            feature = 'RDT_HAT_VAT_3Channel';
        case 10
            feature = 'DT_RT_HAT_VAT_4Channel';
    end
    load(['CNN/',feature,'_CNN_Net'])
    data_path = ['dataset/',test_object,'testset/',feature,'/'];
%% Label generation
N_class = 8;
data_sum = 0;
data_nums = zeros(N_class,1);
for class_choose = 0 : 7
    switch class_choose   
        case 0
            class_str = 'non-gesture';
        case 1
            class_str = 'palm up';
        case 2
            class_str = 'palm down';
        case 3
            class_str = 'palm left';
        case 4
            class_str = 'palm right';
        case 5
            class_str = 'finger double-click';
        case 6
            class_str = 'finger circle';
        case 7
            class_str = 'forefinger-thumb open-close';       
    end
    dataFilefolder = [data_path,class_str,'/'];
    if feat_chose == 10
        dataCount = length(dir([dataFilefolder,'*.mat'])); 
    else
        dataCount = length(dir([dataFilefolder,'*.png'])); 
    end
    data_nums(class_choose+1) = dataCount;
    data_sum = data_sum + dataCount; 
end
    realLabel_str = cell(data_sum,1);
    predLabel_str = cell(data_sum,1);
%% Classification testset
for class_choose = 0 : 7
    switch class_choose      
        case 0
            class_str = 'non-gesture';
        case 1
            class_str = 'palm up';
        case 2
            class_str = 'palm down';
        case 3
            class_str = 'palm left';
        case 4
            class_str = 'palm right';
        case 5
            class_str = 'finger double-click';
        case 6
            class_str = 'finger circle';
        case 7
            class_str = 'forefinger-thumb open-close';  
    end

    dataFilefolder = [data_path,class_str,'/'];
    if feat_chose == 10
        img_all = dir([dataFilefolder,'*.mat']);
    else
        img_all = dir([dataFilefolder,'*.png']);
    end
    
    img_names = {img_all.name};
    dataCount = length(img_all);
for img_ind = 1 : dataCount
    img_name = [dataFilefolder,img_names{img_ind}];
    if feat_chose ~= 10
        image_classify = imread(img_name);
    else
        image_classify = importdata(img_name);
    end
    [label,scores] = classify( GR_CNN_Net , image_classify );    
    class_pred = char(label);

    if class_choose == 0
        realLabel_str{img_ind} = class_str;
        predLabel_str{img_ind} = class_pred;
    else
        count1 = sum(data_nums(1 : class_choose));
        realLabel_str{count1 + img_ind} = class_str;
        predLabel_str{count1 + img_ind} = class_pred;
    end
end %% data index END
end %% class index END

realLabel_str = categorical(realLabel_str);
predLabel_str = categorical(predLabel_str);
%% Plot confusion matrix
figure(1000)
feature_str = strrep(feature,'_','-');
plotconfusion(realLabel_str,predLabel_str,[test_object,'testset ',feature_str,' feature ']);
saveas(gcf,[result_save_path,'Test user ',test_object,' ',feature_str,' classification result.tif'])
end %% feature chose END
end %% user index END