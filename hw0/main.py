import sklearn
import numpy as np
import cv2
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
# For cv2.imwrite:[0,255]; 
# cv2.imshow expects [0,1] for floating point and [0,255] for unsigned chars
# Usage:
# cv2.imshow("window title", array.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()*

trainLst = ["./p1_data/" + str(i) + "_" + str(j) + ".png" for i in range(1,41) for j in range(1,10)]
testLst = ["./p1_data/" + str(i) + "_10.png" for i in range(1,41)]

# training set contains first 9 images of each subject
X_train = np.array([cv2.imread(name,0) for name in trainLst])  # (360, 2576)
y_train = np.array([i for i in range(1,41) for _ in range(9)]) # (360,)

# testing set contains the last image of each subject
X_test = np.array([cv2.imread(name,0) for name in testLst])  # (40, 2576)
y_test = np.array([i for i in range(1,41)])  # (40,)

X_train = X_train.reshape(360,-1)
X_test = X_test.reshape(40,-1)


# * ==============Question 1==============
# mean face
mean_face =  X_train.mean(axis=0)
cv2.imwrite("./results/mean_face.png", mean_face.reshape(56,46))
print("mean face shape: " + str(mean_face.shape)) # (2576,)
print("X_train shape: " + str(X_train.shape)) # (360, 2576)

#pca and first four eigenfaces
pca = PCA().fit(X_train-mean_face)
eigenfaces = pca.components_

# normalize
a = 255/(np.max(eigenfaces, axis=1) - np.min(eigenfaces, axis=1))
a = np.expand_dims(a, -1)
b = -a*np.expand_dims(np.min(eigenfaces,axis=1), -1)
eigenfaces = (eigenfaces*a + b).reshape(360, 56, 46)

def show_and_save(array, title, filename):
    plt.imshow(array, cmap='gray')
    plt.title(title, fontsize=20)
    plt.savefig("./results/"+ filename + ".png")
    plt.show()
    plt.close()

show_and_save(mean_face.reshape(56,46),"mean face", "mean_face")
for i in range(4):
   show_and_save(eigenfaces[i],"eigenface"+ str(i+1) , "eigenface_" + str(i+1))


# * ==============Question 2==============
img = cv2.imread("./p1_data/8_1.png",0).reshape(1,-1)
output = pca.transform(img - mean_face)  # (1,360)

reconstructs = []
lst = [3,50,170,240,345]
for i in lst:
    reconstruct_result = output[:,:i] @ pca.components_[:i] + mean_face  # matrix multiplication 
    reconstructs.append(reconstruct_result.reshape(56,46))

for i in range(5):
    show_and_save(reconstructs[i], "n = " + str(lst[i]), "reconstruct_" + str(i))


# * ====Question 3====
# print(reconstructs[0].shape)  # (56,46)
# print(img.shape)  # (1,2576)
for i in range(len(lst)):
    mse = np.mean((img - reconstructs[i].reshape(1,-1))**2)  # pixel-wise subtraction  
    print("n:{:<3d}, MSE:{:<15f}".format(lst[i], mse))


# * ====Question 4====
kLst = [1,3,5]
nLst = [3,50,170]
train_pca = pca.transform(X_train - mean_face)
#print(train_pca.shape) 
#print(y_train.shape) # (360,)
for k in kLst:
    for n in nLst:
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, train_pca[:,:n], y_train, cv=3, scoring="accuracy")
        print("k={}, n={:<3d}, Acc:{:<15f}".format(k,n,scores.mean()))


# * ====Question 5====
k, n  = 1, 50
test_pca = pca.transform(X_test - mean_face)
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(train_pca[:,:n], y_train)
clf.score(test_pca[:,:n], y_test)

print("Score:", clf.score(test_pca[:,:n], y_test))
