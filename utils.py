import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def split_db_2to1(D, L, seed=1):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def plot_simple_hist(DP, L, title = "LDA Fingerprint Projection"):
    
    dp_flat = np.array(DP).flatten()
    
    l_flat = np.array(L).flatten()

    print(f"Original DP shape: {DP.shape}")
    print(f"Flattened DP shape: {dp_flat.shape}")

    class0_pts = dp_flat[l_flat == 0]
    class1_pts = dp_flat[l_flat == 1]

    plt.figure(figsize=(8, 5))
    
   
    plt.hist(class0_pts, bins=5, density=True, alpha=0.4, 
             color='sandybrown', label='Non-Genuine (0)', 
             edgecolor='peru', linewidth=0.5)

    #
    plt.hist(class1_pts, bins=5, density=True, alpha=0.4, 
             color='forestgreen', label='Genuine (1)', 
             edgecolor='seagreen', linewidth=0.5)

    
    plt.title(title, y=-0.2, fontsize=14, fontfamily='serif') 
    plt.legend(loc='upper right', frameon=True)
    
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def load(data_path):
    with open(data_path) as f:
        data = f.read().splitlines()
    
    filtered_data = []
    L = []
    for row in data:
        wanted_data = [row.split(",")[i] for i in range(6)]
        filtered_data.append(wanted_data)
        L.append(int(row.split(", ")[6]))
    
    matrix_data = np.array(filtered_data, dtype=float)
    D = matrix_data.T
    L = np.array(L)
    return D, L


def plot_correlation_matrix(D, feature_names=None, figsize=(8, 6), cmap='coolwarm'):
    
    if D.shape[0] < D.shape[1]:
        data_matrix = D.T
    else:
        data_matrix = D

    num_features = data_matrix.shape[1]
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(num_features)]

    df = pd.DataFrame(data_matrix, columns=feature_names)
    corr_matrix = df.corr(method='pearson')

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix, 
        annot=True,            
        fmt=".2f",             
        cmap=cmap,             
        vmin=-1.0, vmax=1.0,   
        square=True, 
        linewidths=0.5,
        cbar_kws={"shrink": .8}
    )

    plt.title("Feature Correlation Matrix (Pearson)", fontsize=14, pad=12)
    plt.tight_layout()
    plt.show()

    return corr_matrix

def plot_feature_distributions(D, L, feature_names=None, class_names=None, bins=15):
   
    num_features, num_samples = D.shape
    
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(num_features)]
        
    df = pd.DataFrame(D.T, columns=feature_names)
    
    if class_names is not None:
        df['Label'] = [class_names.get(val, str(val)) for val in L]
    else:
        df['Label'] = L.astype(str)

    ncols = 2
    nrows = int(np.ceil(num_features / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()

    for i in range(num_features):
        feat = feature_names[i]
        sns.histplot(
            data=df,
            x=feat,
            hue='Label',
            kde=True,               
            bins=bins,
            stat="density",          
            common_norm=False,
            element="step",
            palette="Set1",
            ax=axes[i]
        )
        axes[i].set_title(f"Distribution of {feat}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Density")

    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Feature Class Distributions & Modality Analysis", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.show()

# D, L = load("data/trainData.txt")
# # plot_correlation_matrix(D)
# classes = {0: "Class 0", 1: "Class 1"}

# # Call function
# plot_feature_distributions(D, L, class_names=classes, bins=20)