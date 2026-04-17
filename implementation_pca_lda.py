import numpy as np
import matplotlib.pyplot as plt
from lda import train_lda

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

def make_histograms_pca(P, L):
    labels = np.array(L).flatten()
    unique_labels = np.unique(labels)
    
    # Get the number of components (rows) in the projection matrix P
    num_components = P.shape[0] 
    
    # Set up the grid (e.g., if you have 6 components, it makes a 3x2 grid)
    cols = 2
    rows = (num_components + 1) // cols
    fig = plt.figure(figsize=(12, 4 * rows))
    fig.suptitle('PCA Component Distributions: Live vs Spoof', fontsize=16)

    for i in range(num_components):
        plt.subplot(rows, cols, i + 1)
        
        for label in unique_labels:
            # We use labels here to "unmix" the data for the plot
            class_data = P[i, labels == label]
            
            # Using different colors for each class
            color = 'blue' if label == 0 else 'red'
            name = 'Spoof' if label == 0 else 'Geniue'
            
            plt.hist(class_data, bins=40, alpha=0.5, 
                     label=f'Class {label} ({name})', color=color, density=True)
        
        plt.title(f'Principal Component {i+1}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Adjust layout so titles don't overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def calculate_mean(D):
    return D.mean(1)

def calculate_centered_data(D, mu):
    mu = mu.reshape(mu.size, 1)
    return D - mu

def calculate_covariance(D, DC):
    return (DC @ DC.T) / float(D.shape[1])

def apply_eigen(C):
    eigenValues, eigenVectors = np.linalg.eigh(C)
    print("Eigen values: s: ", eigenValues)
    print("Eigen vector: ", eigenVectors)
    return eigenValues, eigenVectors  

def eigen_Vector_Backwards(eigenVector, limit = 4):
    return eigenVector[:, ::-1][:, 0:limit]  

def make_projection(eigenVectors, D):
    return np.dot(eigenVectors.T, D)


def train_pca(D, L, components = 2):
    mu = calculate_mean(D)
    print("Mean:", mu)
    DC = calculate_centered_data(D, mu)
    #print(DC)
    C = calculate_covariance(D, DC)
    #print(C)
    eigenValues, eigenVectors = apply_eigen(C)
    # print(eigenValues)
    # print(eigenVectors)
    eigenVectors = eigen_Vector_Backwards(eigenVectors, components)
    #print(eigenVectors)
    DP = make_projection(eigenVectors, D)
    make_histograms_pca(DP, L)
    
    return DP, eigenVectors

def plot_simple_hist(DP, L, title = "LDA Fingerprint Projection"):
#    Force DP to be 1D
    
    dp_flat = np.array(DP).flatten()
    
    # Force Labels to be 1D just in case
    l_flat = np.array(L).flatten()

    print(f"Original DP shape: {DP.shape}")
    print(f"Flattened DP shape: {dp_flat.shape}")

    # 3. Separate the points
    class0_pts = dp_flat[l_flat == 0]
    class1_pts = dp_flat[l_flat == 1]

    plt.figure(figsize=(8, 5))
    
    # 4. Plot using the 1D arrays
   
    plt.hist(class0_pts, bins=5, density=True, alpha=0.4, 
             color='sandybrown', label='Non-Genuine (0)', 
             edgecolor='peru', linewidth=0.5)

    #
    plt.hist(class1_pts, bins=5, density=True, alpha=0.4, 
             color='forestgreen', label='Genuine (1)', 
             edgecolor='seagreen', linewidth=0.5)

    
    plt.title(title, y=-0.2, fontsize=14, fontfamily='serif') # Moves title to bottom
    plt.legend(loc='upper right', frameon=True)
    
    # Standard labels
    plt.ylabel("") # The example image doesn't label the y-axis
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    D, L = load("FingerPrintSpoofingDetection/data/trainData.txt")
    print(D.shape)
    DP, W = train_lda(D, L, 2, 1)
    #plot_simple_hist(DP, L)    
    print("DP: \n")
    print(DP)

    print("W: \n")
    print(W)



    # mu = calculate_mean(D)
    # #print(mu)
    # DC = calculate_centered_data(D, mu)
    # #print(DC)
    # C = calculate_covariance(D, DC)
    # #print(C)
    # eigenValues, eigenVectors = apply_eigen(C)
    # # print(eigenValues)
    # # print(eigenVectors)
    # eigenVectors = eigen_Vector_Backwards(eigenVectors)
    # #print(eigenVectors)
    # P = make_projection(eigenVectors, D)
    # make_histograms_pca(P, L)
    