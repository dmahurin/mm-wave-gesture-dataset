import numpy as np
import os
import matplotlib.pyplot as plt
##
for user in [1,2]:
    match user:
        case 1:
            test_object = 'A'
        case 2:
            test_object = 'B'

    result_save_path = 'results/' + test_object + 'testset/'
    if os.path.exists(result_save_path) == 0:
        os.mkdir(result_save_path)
    for feat_chose in np.arange(1,11):
        match feat_chose:
            case 1:
                feature = 'RDTI'
            case 2:
                feature = 'DTI'
            case 3:
                feature = 'RTI'
            case 4:
                feature = 'HATI'
            case 5:
                feature = 'VATI'
            case 6:
                feature = 'HVTI'
            case 7:
                feature = 'DT_HAT_VAT_3Channel'
            case 8:
                feature = 'DT_RT_HVT_3Channel'
            case 9:
                feature = 'RDT_HAT_VAT_3Channel'
            case 10:
                feature = 'DT_RT_HAT_VAT_4Channel'

        scipy.io.loadmat(np.array(['CNN/',feature,'_CNN_Net']))
        data_path = np.array(['dataset/',test_object,'testset/',feature,'/'])
        ## Label generation
        N_class = 8
        data_sum = 0
        data_nums = np.zeros((N_class,1), dtype=int)
        for class_choose in np.arange(0,8):
            match class_choose:
                case 0:
                    class_str = 'non-gesture'
                case 1:
                    class_str = 'palm up'
                case 2:
                    class_str = 'palm down'
                case 3:
                    class_str = 'palm left'
                case 4:
                    class_str = 'palm right'
                case 5:
                    class_str = 'finger double-click'
                case 6:
                    class_str = 'finger circle'
                case 7:
                    class_str = 'forefinger-thumb open-close'
            dataFilefolder = np.array([data_path,class_str,'/'])
            if feat_chose == 10:
                dataCount = len(dir(np.array([dataFilefolder,'*.mat'])))
            else:
                dataCount = len(dir(np.array([dataFilefolder,'*.png'])))
            data_nums[class_choose + 1] = dataCount
            data_sum = data_sum + dataCount
        realLabel_str = cell(data_sum,1)
        predLabel_str = cell(data_sum,1)
        ## Classification testset
        for class_choose in np.arange(0,8):
            match class_choose:
                case 0:
                    class_str = 'non-gesture'
                case 1:
                    class_str = 'palm up'
                case 2:
                    class_str = 'palm down'
                case 3:
                    class_str = 'palm left'
                case 4:
                    class_str = 'palm right'
                case 5:
                    class_str = 'finger double-click'
                case 6:
                    class_str = 'finger circle'
                case 7:
                    class_str = 'forefinger-thumb open-close'
            dataFilefolder = np.array([data_path,class_str,'/'])
            if feat_chose == 10:
                img_all = dir(np.array([dataFilefolder,'*.mat']))
            else:
                img_all = dir(np.array([dataFilefolder,'*.png']))
            img_names = np.array([img_all.name])
            dataCount = len(img_all)
            for img_ind in np.arange(1,dataCount+1).reshape(-1):
                img_name = np.array([dataFilefolder,img_names[img_ind]])
                if feat_chose != 10:
                    image_classify = imread(img_name)
                else:
                    image_classify = importdata(img_name)
                label,scores = classify(GR_CNN_Net,image_classify)
                class_pred = char(label)
                if class_choose == 0:
                    realLabel_str[img_ind] = class_str
                    predLabel_str[img_ind] = class_pred
                else:
                    count1 = sum(data_nums(np.arange(1,class_choose+1)))
                    realLabel_str[count1 + img_ind] = class_str
                    predLabel_str[count1 + img_ind] = class_pred
        realLabel_str = categorical(realLabel_str)
        predLabel_str = categorical(predLabel_str)
        ## Plot confusion matrix
        plt.figure(1000)
        feature_str = feature.replace('_','-')
        plotconfusion(realLabel_str,predLabel_str,np.array([test_object,'testset ',feature_str,' feature ']))
        saveas(gcf,np.array([result_save_path,'Test user ',test_object,' ',feature_str,' classification result.tif']))

