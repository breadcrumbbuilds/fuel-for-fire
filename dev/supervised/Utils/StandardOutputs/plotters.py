

def plot_confusion_matrix(confmat, filename, classes, n_classes=2, title="Confusion Matrix"):
    plt.title(f"{title}")
    plt.matshow(confmat, cmap=plt.cm.Blues, alpha=0.5)
    plt.gcf().subplots_adjust(left=.5)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            plt.text(x=j, y=i,
                    s=round(confmat[i,j],3), fontsize=6, horizontalalignment='center')
    plt.xticks(np.arange(n_classes), labels=classes)
    plt.yticks(np.arange(n_classes), labels=classes)
    plt.tick_params('both', labelsize=8, labelrotation=45)
    plt.xlabel('predicted label')
    plt.ylabel('reference label', rotation=90)
    plt.savefig(f'{fn}')
    print(f'+w {fn}')
    plt.close()