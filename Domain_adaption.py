from sklearn.decomposition import PCA 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Open human and mouse data
human_df = pd.read_csv("/home/rehan/Documents/sainbiose/MALBOT MOUSE/Norbert_humain/Trab_Human.csv")
mouse_df = pd.read_csv("/home/rehan/Images/LR_HR_2D/Train_Label_7p_lrhr.csv")
#human_df = human_df.drop(range(100,200))
#mouse_df = mouse_df.sample(n=300,random_state=42)
human_df = human_df.drop(columns=['File name'])
mouse_df = mouse_df.drop(columns=['File name'])

#mouse_df= mouse_df.drop(['Euler number','Trabecular thickness (plate model)','Trabecular pattern factor','Average object area'],axis=1)
#human_df= human_df.drop(['Euler number','Trabecular thickness (plate model)','Trabecular pattern factor','Average object area'],axis=1)

# rescale the data
scaler = StandardScaler()
scaler2 = StandardScaler()

H_scaled = scaler.fit_transform(human_df[:-100])
M_scaled = scaler2.fit_transform(mouse_df)

# Perform PCA on both Source and Target (Human and Mouse)
pca_m = PCA(n_components=3)
pca_h = PCA(n_components=3)


X_M = pca_m.fit_transform(M_scaled.T)
X_H = pca_h.fit_transform(H_scaled.T)

for i in range(X_M.shape[1]):
    X_M[:,i] = X_M[:,i] / np.linalg.norm(X_M[:,i])
    X_H[:,i] = X_H[:,i] / np.linalg.norm(X_H[:,i])

print(X_M @ X_M.T)
print(X_H @ X_H.T)
H_m = H_scaled @ (X_H @ X_H.T @ X_M @ X_M.T)

H_m_rescale = scaler2.inverse_transform(H_m)
pd.DataFrame(H_m).to_csv("/home/rehan/Documents/Algorithm/BPNN/H_m.csv")
