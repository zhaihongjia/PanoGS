import numpy as np
from sklearn import cluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch as t
import os
import cv2
from PIL import Image

def calc_pca(feature, dims=3):
    '''
    feature: tensor, [H, W, D]
    
    return
    rgb: (-1, 1)?
    '''
    # feature = t.from_numpy(feature)
    feature = t.nn.functional.normalize(feature, p=2, dim=-1)

    print('PCA feature: ', feature.shape)

    X = feature.flatten(0, -2).cpu().numpy()
    pca = PCA(n_components=dims)
    pca.fit(X)
    X_rgb = pca.transform(X).reshape(*feature.shape[:2], dims)

    # scale to [0, 255]
    X_rgb = X_rgb - np.min(X_rgb)
    X_rgb = X_rgb / np.max(X_rgb)
    X_rgb = np.uint8(255 * X_rgb)
    return X_rgb

def point_cloud_feature_pca(feature, dims=3):
    '''
    feature: tensor, [N, D]
    
    return
    rgb: (-1, 1)?
    '''
    feature = t.nn.functional.normalize(feature, p=2, dim=-1)
    print('PCA feature: ', feature.shape)

    pca = PCA(n_components=dims)
    pca.fit(feature.cpu().numpy())
    X_rgb = pca.transform(feature.cpu().numpy()).reshape(*feature.shape[:2], dims)

    # scale to [0, 255]
    X_rgb = X_rgb - np.min(X_rgb)
    X_rgb = X_rgb / np.max(X_rgb)
    X_rgb = np.uint8(255 * X_rgb)
    return X_rgb


