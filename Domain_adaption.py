from sklearn.decomposition import PCA 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Open human and mouse data
human_df = pd.read_csv("/home/rehan/Documents/sainbiose/MALBOT MOUSE/Norbert_humain/Trab_Human.csv")
mouse_df = pd.read_csv("/home/rehan/Images/LR_HR_2D/Train_Label_7p_lrhr.csv")
human_df = human_df.drop(range(100,200))
mouse_df = mouse_df.sample(n=300,random_state=42)
human_df = human_df.drop(columns=['File name'])
mouse_df = mouse_df.drop(columns=['File name'])

# rescale the data
scaler = StandardScaler()
scaler2 = StandardScaler()

H_scaled = scaler.fit_transform(human_df)
M_scaled = scaler2.fit_transform(mouse_df)

# Perform PCA on both Source and Target (Human and Mouse)
pca_m = PCA(n_components=3)
pca_h = PCA(n_components=3)

pca_m.fit(M_scaled)
pca_h.fit(H_scaled)

X_M = pca_m.transform(M_scaled)
X_H = pca_h.transform(H_scaled)

H_m = H_scaled.T @ (X_H @ X_H.T @ X_M @ X_M.T)

#pd.DataFrame(H_m)
H_m_rescale = scaler2.inverse_transform(H_m)
#.to_csv("/home/rehan/Documents/sainbiose")
