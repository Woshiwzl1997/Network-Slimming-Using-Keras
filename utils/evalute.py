import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_and_evalute(name,compact_model ,X_test ,Y_test ,history,snrs,all_label,test_idx,M_old):

    score = compact_model.evaluate(X_test, Y_test, verbose=0, batch_size=1024)
    print("模型loss:{},acc:{}：".format(score[0],score[1]))
    # Show loss curves
    plt.figure()
    plt.title(name+'Training performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()
    plt.show()

    # Plot confusion matrix
    # plt.figure()
    # test_Y_hat = model.predict(X_test, batch_size=batch_size)
    # conf = np.zeros([len(M_old), len(M_old)])
    # confnorm = np.zeros([len(M_old), len(M_old)])
    # for i in range(0, X_test.shape[0]):
    #     j = list(Y_test[i, :]).index(1)
    #     k = int(np.argmax(test_Y_hat[i, :]))
    #     conf[j, k] = conf[j, k] + 1
    # for i in range(0, len(M_old)):
    #     confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    # plot_confusion_matrix(confnorm, labels=M_old)

    # Plot confusion matrix of different SNRS
    acc = {}
    for snr in snrs[0:len(snrs):3]:

        # extract classes @ SNR
        test_SNRs = list(map(lambda x: all_label[x][1], test_idx))
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        # estimate classes
        test_Y_i_hat = compact_model.predict(test_X_i)
        conf = np.zeros([len(M_old), len(M_old)])
        confnorm = np.zeros([len(M_old), len(M_old)])
        for i in range(0, test_X_i.shape[0]):
            j = list(test_Y_i[i, :]).index(1)
            k = int(np.argmax(test_Y_i_hat[i, :]))
            conf[j, k] = conf[j, k] + 1
        for i in range(0, len(M_old)):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
        # plt.figure()
        # plot_confusion_matrix(confnorm, labels=M_old, title="ConvNet Confusion Matrix (SNR=%d)" % (snr))

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print("Overall Accuracy: ", cor / (cor + ncor))
        acc[snr] = 1.0 * cor / (cor + ncor)

    # Plot accuracy curve
    plt.figure()
    plt.plot(snrs[0:len(snrs):3], list(map(lambda x: acc[x], snrs[0:len(snrs):3])))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title(name+"CNN2 Classification Accuracy on RadioML 2016.10 Alpha")