from sklearn import metrics
import matplotlib.pyplot as plt

def ReportCard(y_true, y_pred, y_proba):
    #%matplotlib inline
    #print('\nReport for:',name)
    target_names = ['low', 'high']
    report = metrics.classification_report(y_true, y_pred,
                                           target_names=target_names)
    confMat = metrics.confusion_matrix(y_true,y_pred)
    f1Score = metrics.f1_score(y_true,y_pred)
    acc = metrics.accuracy_score(y_true,y_pred)
    logLoss = metrics.log_loss(y_true,y_pred)
    auc = metrics.roc_auc_score(y_true,y_proba[:,1])
    fpr,tpr,thresholds = metrics.roc_curve(y_true,y_proba[:,1])

    return {'report':report, "confusion_matrix":confMat, "f1_score":f1Score,
            'accuracy':acc, 'log_loss':logLoss, 'auc':auc,
            'ftt':[fpr,tpr,thresholds]}

def PlotReport(ftt,name):
    """ Used to Generate a Plot from Results of the other function
    """
    fpr,tpr,thresholds = ftt
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for {}'.format(name))
    plt.legend(loc='best')
    plt.show()
