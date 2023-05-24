import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import numpy as np
from matplotlib import cm
def get_confusionmatrix(filename):
    title= get_title(filename)
    df = pd.read_pickle(filename)
    tp = float(df["test_n_true_positive"].iloc[-1])
    fp = float(df["test_n_positive"].iloc[-1])-float(df["test_n_true_positive"].iloc[-1])
    fn = float(df["test_n_negative"].iloc[-1]) - float(df["test_n_true_negative"].iloc[-1])
    tn = float(df["test_n_true_negative"].iloc[-1])
    cf = np.array([[tn,fp],
            [fn,tp]], dtype=np.float32)
    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        cf.flatten()/np.sum(cf)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(cf, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Confusion Matrix\n'+title+ '\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    #plt.show()
    plt.savefig(filename.replace(".pkl","CF.png"))
    plt.close()
    print(cf)
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
def get_all_summary(filename):
    df = pd.read_pickle(filename)
    title = get_title(filename)
    train_accuracy = 1-df["train_misclass"].iloc[-1]
    train_f1 = (df["train_F1"].iloc[-1])
    train_auc = (df["train_auc"].iloc[-1])
    train_loss =  (df["train_loss"].iloc[-1])
    valid_accuracy = 1-df["valid_misclass"].iloc[-1]
    valid_f1 = (df["valid_F1"].iloc[-1])
    valid_auc = (df["valid_auc"].iloc[-1])
    valid_loss =  (df["valid_loss"].iloc[-1])
    test_accuracy = 1-df["test_misclass"].iloc[-1]
    test_f1 = (df["test_F1"].iloc[-1])
    test_auc = (df["test_auc"].iloc[-1])
    test_loss =  (df["test_loss"].iloc[-1])
    return [title,train_accuracy,train_f1,train_auc,train_loss,valid_accuracy,valid_f1,valid_auc,valid_loss,test_accuracy,test_f1,test_auc,test_loss]
def get_test_summary(filename,ext="test"):
    df = pd.read_pickle(filename)
    title = get_title(filename)
    accuracy = 1-df[ext+"_misclass"].iloc[-1]
    f1 = (df[ext+"_F1"].iloc[-1])
    auc = (df[ext+"_auc"].iloc[-1])
    loss =  (df[ext+"_loss"].iloc[-1])
    return [title,accuracy,f1,auc,loss]

def get_title(filename):
    title=""
    if "noclean" in filename:
        if "deep+eegnet" in filename:
            title = "Deep + EEGNet without Line Noise Removed"
        elif "shallow+eegnet" in filename:
            title = "Shallow + EEGNet without Line Noise Removed"
        elif "eegnet" in filename:
            title = "EEGNet without Line Noise Removed"
        elif "shallow" in filename:
            title = "ShallowNet without Line Noise Removed"
        elif "deep" in filename:
            title = "DeepConvNet without Line Noise Removed"
    else:
        if "deep+eegnet" in filename:
            title = "Deep + EEGNet with Line Noise Removed"
        elif "shallow+eegnet" in filename:
            title = "Shallow + EEGNet with Line Noise Removed"
        elif "eegnet" in filename:
            title = "EEGNet with Line Noise Removed"
        elif "shallow" in filename:
            title = "ShallowNet with Line Noise Removed"
        elif "deep" in filename:
            title = "DeepConvNet with Line Noise Removed"
    return title
def read_all_file_names(path, extension, key="time"):

    file_paths = glob.glob(path + '**/*' + extension, recursive=True)
    return file_paths     #RETURN TO STOP SORTING

    if key == 'time':
        return sorted(file_paths, key=time_key)

    elif key == 'natural':
        return sorted(file_paths, key=natural_key)
def plot_loss(filename):
    
    df = pd.read_pickle(filename)

    df.train_loss = smooth(df.train_loss,3)
    df.valid_loss = smooth(df.valid_loss,3)
    df = df.iloc[:-1 , :]
    df.plot(y=["train_loss","valid_loss"],colormap = cm.get_cmap('bwr'))

    ax = plt.gca()
    
    ax.set_ylim([0.3, 1])
    title=get_title(filename)
    
    plt.title(title)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(filename.replace(".pkl","loss.png"))
    plt.close()
def plot_f1(filename):
    
    df = pd.read_pickle(filename)
    df.train_F1 = smooth(df.train_F1,3)
    df.valid_F1 = smooth(df.valid_F1,3)
    df = df.iloc[:-1 , :]
    df.plot(y=["train_F1","valid_F1"],colormap = cm.get_cmap('jet'))

    ax = plt.gca()
    ax.set_ylim([0.5, 1])
    title=get_title(filename)
    
    plt.title(title)
    
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.savefig(filename.replace(".pkl","F1.png"))
    plt.close()
def plot_accuracy(filename):
    
    df = pd.read_pickle(filename)
    train_accuracy = 1-df.train_misclass
    valid_accuracy = 1-df.valid_misclass
    test_accuracy = 1-df.test_misclass
    df['train_accuracy'] = train_accuracy
    df['valid_accuracy'] = valid_accuracy
    df['test_accuracy'] = test_accuracy
    print(filename)
    print("Epoch " + str(len(df.index)))
    #print( "Train Accuracy :" + (df["train_accuracy"].iloc[-1]).astype('str'))
    #print( "Valid Accuracy :" + (df["valid_accuracy"].iloc[-1]).astype('str'))
    #print( "Test Accuracy :" + (df["test_accuracy"].iloc[-1]).astype('str'))
    #print( "Train F1 :" + (df["train_F1"].iloc[-1]).astype('str'))
    #print( "Valid F1 :" + (df["valid_F1"].iloc[-1]).astype('str'))
    #print( "Test F1 :" + (df["test_F1"].iloc[-1]).astype('str'))
    #print( "Train AUC :" + (df["train_auc"].iloc[-1]).astype('str'))
    #print( "Valid AUC :" + (df["valid_auc"].iloc[-1]).astype('str'))
    #print( "Test AUC :" + (df["test_auc"].iloc[-1]).astype('str'))
    df.train_accuracy = smooth(df.train_accuracy,3)
    df.valid_accuracy = smooth(df.valid_accuracy,3)
    df = df.iloc[:-1 , :]
    df.plot(y=["train_accuracy","valid_accuracy"],lw =2)

    ax = plt.gca()
    ax.set_ylim([0.5, 0.9])
    title=get_title(filename)
   
    plt.title(title)
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(filename.replace(".pkl","accuracy.png"))
    plt.close()
    #plt.show()
#print("test")
folderpath = "C:\\Users\\randy\\Desktop\\tes2"
filename = read_all_file_names(folderpath,".pkl")
summary =[]
for i in filename:
    summary.append(get_all_summary(i))
    get_confusionmatrix(i)
    plot_loss(i)
    plot_f1(i)
    plot_accuracy(i)
    
df = pd.DataFrame(summary, columns=['Title', 'Train Accuracy','Train F1','Train AUC','Train Loss', 'Valid Accuracy','Valid F1','Valid AUC','Valid Loss', 'Test Accuracy','Test F1','Test AUC','Test Loss'])
print(df)
df.to_excel("C:\\Users\\randy\\Desktop\\tes2\\summary.xlsx")