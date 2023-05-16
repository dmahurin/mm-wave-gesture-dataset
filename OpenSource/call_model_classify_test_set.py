import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io
import imageio.v3 as imageio
import onnxruntime
#from sklearn.metrics import confusion_matrix

class_strings = np.array(['non-gesture','palm up','palm down','palm left','palm right','finger double-click','finger circle','forefinger-thumb open-close'])

def classify(net_session, img):
    img = np.array(img).astype(np.float32)

    if len(img.shape) > 2:
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
    else:
        img = np.expand_dims(np.expand_dims(img, axis=0), axis=1)

    predictions = net_session.run(None, {net_session.get_inputs()[0].name: img})[0]
    predicted_label = np.argmax(predictions)

    return predicted_label, predictions

def plotconfusion(actual, predicted, class_strings, title):
    nclasses = len(class_strings)
    actual = np.array(actual)
    predicted = np.array(predicted)
    # cm = confusion_matrix(actual, predicted)
    cm = np.bincount(nclasses *  predicted + actual, minlength=nclasses ** 2).reshape(nclasses, nclasses)

    plt.title(title + ' Confusion Matrix')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    cm = np.transpose(cm)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.xticks(np.arange(nclasses), class_strings, rotation=-45, ha='left')
    plt.yticks(np.arange(nclasses), class_strings)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

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

        net_session = onnxruntime.InferenceSession('CNN/' + feature + '_CNN_Net' + '.onnx')
        data_path = 'dataset/' + test_object + 'testset/' + feature + '/'
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

            dataFilefolder = data_path + class_str + '/'
            if feat_chose == 10:
                dataCount = len([ file for file in os.listdir(dataFilefolder) if file.endswith('.mat')])
            else:
                dataCount = len([ file for file in os.listdir(dataFilefolder) if file.endswith('.png')])
            data_nums[class_choose] = dataCount
            data_sum = data_sum + dataCount
        realLabel = [None] * data_sum
        predLabel = [None] * data_sum
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
            dataFilefolder = data_path + class_str + '/'
            if feat_chose == 10:
                img_all = [ file for file in os.listdir(dataFilefolder) if file.endswith('.mat')]
            else:
                img_all = [ file for file in os.listdir(dataFilefolder) if file.endswith('.png')]
            img_names = img_all
            dataCount = len(img_all)
            for img_ind in np.arange(0,dataCount):
                img_name = dataFilefolder + img_names[img_ind]
                if feat_chose != 10:
                    image_classify = imageio.imread(img_name)
                else:
                    image_classify =  scipy.io.loadmat(img_name)
                    image_classify =  image_classify['img_3channels']
                label,scores = classify(net_session,image_classify)

                class_from_model_index = np.argsort(class_strings, axis=0)
                if class_choose == 0:
                    realLabel[img_ind] = class_choose
                    predLabel[img_ind] = class_from_model_index[label]

                else:
                    count1 = sum(data_nums[0:class_choose])
                    realLabel[int(count1 + img_ind)] = class_choose
                    predLabel[int(count1 + img_ind)] = class_from_model_index[label]

        ## Plot confusion matrix
        feature_str = feature.replace('_','-')
        plotconfusion(realLabel, predLabel, class_strings, test_object + 'testset ' + feature_str + ' feature ')
        plt.pause(3)
        plt.clf()

