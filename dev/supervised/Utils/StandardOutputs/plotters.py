

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
    print(f'+w {filename}')
    plt.close()


def imshow_ref_vs_predict_multiclass(y, pred, filename, img_shape, classes, title="Reference vs Prediction"):
    f, ax = plt.subplots(2,1, sharey=True, figsize=(30,15))
    f.suptitle(title)
    colormap = ax[0].imshow(y.reshape(img_shape), cmap='cubehelix', vmin=0, vmax=len(classes)+1)
    ax[0].set_title('Ground Reference')
    ax[1].imshow(pred.reshape(sub_img), cmap='cubehelix', vmin=0, vmax=len(classes)+1)
    ax[1].set_title('Prediction')

    cbar = f.colorbar(colormap,
                        ax=ax.ravel().tolist(),
                        orientation='vertical',
                        boundaries=range(len(classes) + 1), shrink=.95,extend='max', extendrect=True, drawedges=True, spacing='uniform')
    cbar.ax.set_yticklabels(classes,fontsize=20)
    plt.savefig(f'{filename}')
    print(f'+w {filename}')
    plt.close()

def imshow_with_colorbar(y, filename, img_shape, classes, title="Image"):
    plt.title('Test Reference')
    colormap = plt.imshow(y.reshape(img_shape), cmap='cubehelix')
    cbar = plt.colorbar(colormap,
                    orientation='vertical',
                    boundaries=range(len(classes) + 1), shrink=.95,extend='max', extendrect=True, drawedges=True, spacing='uniform')
    cbar.ax.set_yticklabels(classes)
    plt.savefig(f'{filename}')
    print(f'+w {filename}')
    plt.close()