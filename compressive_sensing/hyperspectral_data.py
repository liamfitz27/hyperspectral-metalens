#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import os
from tqdm import tqdm
from sklearn.decomposition import DictionaryLearning, SparseCoder
#%%

path = r"/Users/liam/Downloads/hyspecnet-11k/patches/"

imgs_l = []
for folder1 in tqdm(os.listdir(path)):
    if folder1.startswith("."):
        continue
    dir1 = os.path.join(path, folder1)
    for folder2 in os.listdir(dir1):
        if folder2.startswith("."): 
            continue
        dir2 = os.path.join(dir1, folder2)
        for file in os.listdir(dir2):
            if file.endswith("SPECTRAL_IMAGE.TIF"):
                img_dir = os.path.join(dir2, file)
                img = tiff.imread(img_dir)
                img = np.array(img)
                imgs_l.append(img)

imgs_band_sample = []
for img in imgs_l:
    img_band_sample = []
    for i in range(np.shape(img)[0]):
        img_band_sample.append(img[i])
    imgs_band_sample.append(img_band_sample)
imgs = np.array(imgs_band_sample)[:, :-2, :100, :100]

N_img = len(imgs)
N_wvl = np.shape(imgs)[1]
N_px = np.shape(imgs)[2]
imgs_flat = np.resize(imgs, (len(imgs), N_wvl*N_px**2))
print("Number of images: ", len(imgs))
print("Number of bands: ", imgs[0].shape[0])
print("Image size: ", (N_px, N_px))

#%%
spect = []
for img in imgs:
    for j in range(img.shape[1]):
        for k in range(img.shape[2]):
            spect.append(img[:, j, k])

#%%
# PLOT RANDOM SPECT
n = np.random.choice(len(spect))
plt.plot(spect[n][167:])

#%%
# PLOT RANDOM IMG
n = np.random.choice(len(imgs))
band = 0
img = np.resize(imgs[n, band], (N_px, N_px))
plt.imshow(img)

# %%
# DICTIONARY LEARNING 
dict_learner = DictionaryLearning(
    n_components=50, transform_alpha=0.01, fit_algorithm="cd", transform_algorithm="lasso_cd",
    random_state=42, verbose=True, max_iter=100, transform_max_iter=10000, positive_dict=False, positive_code=False,
)

sparse_coeffs = dict_learner.fit(imgs_flat).transform(imgs_flat)
sparse_dict = dict_learner.components_
sparse_reps = sparse_coeffs@sparse_dict

filename = r"hyspecnet11k_dictlearning.npz"
np.savez(filename, sparse_coeffs=sparse_coeffs, sparse_dict=sparse_dict)

#%%
# LOAD/SHOW DICTIONARY REPRESENTATIONS
dict_learned = np.load(filename)

sparse_coeffs = dict_learned["sparse_coeffs"]
sparse_dict = dict_learned["sparse_dict"]
sparse_reps = sparse_coeffs @ sparse_dict

i = np.random.randint(len(imgs_flat))
band = 2
img0 = np.reshape(imgs_flat[i], (N_wvl, N_px, N_px))
img0 = img0[band]

img1 = np.reshape((sparse_reps[i,:]/np.max(sparse_reps[i,:])*255).astype(np.uint8), (N_wvl, N_px, N_px))
img1 = img1[band]
fig, ax = plt.subplots(1, 2, dpi=200)
ax[0].imshow(img0)
ax[1].imshow(img1)

#%%
# LOAD VALIDATION IMAGES
path = r'/Users/liam/Downloads/hyspecnet-11k_2/patches'

imgs_v = []
for folder1 in tqdm(os.listdir(path)):
    if folder1.startswith("."):
        continue
    dir1 = os.path.join(path, folder1)
    for folder2 in os.listdir(dir1):
        if folder2.startswith("."): 
            continue
        dir2 = os.path.join(dir1, folder2)
        for file in os.listdir(dir2):
            if file.endswith("SPECTRAL_IMAGE.TIF"):
                img_dir = os.path.join(dir2, file)
                img = tiff.imread(img_dir)
                imgs_v.append(img)

imgs_v_band_sample = []
for img in imgs_v:
    img_band_sample = []
    for i in range(np.shape(img)[0]):
        if i % 50 == 0:
            img_band_sample.append(img[i])
    imgs_v_band_sample.append(img_band_sample)
imgs_v = np.array(imgs_v_band_sample)[:, :-2, :100, :100]


N_img = len(imgs_v)
N_wvl = imgs_v[0].shape[0]
N_px = imgs_v[0].shape[1]
imgs_v_flat = np.resize(imgs_v, (len(imgs_v), N_wvl*N_px**2))
print("Number of images: ", len(imgs_v))
print("Number of bands: ", imgs_v[0].shape[0])
print("Image size: ", (N_px, N_px))

#%%
# DECODE VALIDATION IMAGES
filename = r"hyspecnet11k_dictlearning.npz"
dict_learned = np.load(filename)
sparse_dict = dict_learned["sparse_dict"]

coder = SparseCoder(
    dictionary=sparse_dict , transform_algorithm='lasso_cd',
    transform_alpha=0.01, positive_code=False,
)

sparse_coeffs = coder.transform(imgs_v_flat)
sparse_reps = sparse_coeffs@sparse_dict

#%%
# SHOW VALIDATION IMG AT SPECIFIC BAND
i = np.random.randint(len(imgs_v_flat))
band = 0
img0 = np.reshape(imgs_v_flat[i], (N_wvl, N_px, N_px))
img0 = img0[band]
img1 = np.reshape((sparse_reps[i,:]/np.max(sparse_reps[i,:])*255).astype(np.uint8), (N_wvl, N_px, N_px))
img1 = img1[band]

fig, ax = plt.subplots(1, 2, dpi=200)
ax[0].imshow(img0)
ax[1].imshow(img1)

#%%
# SHOW VALIDATION IMG SPECTRUM AT SPECIFIC PIXEL
i = np.random.randint(len(imgs_v_flat))
idx, idy = np.random.randint(N_px), np.random.randint(N_px)
img = np.reshape(imgs_v_flat[i], (N_wvl, N_px, N_px))
spectrum = img[:, idx, idy]

img_sparse = np.reshape(sparse_reps[i], (N_wvl, N_px, N_px))
spectrum_sparse = img_sparse[:, idx, idy]
plt.plot(spectrum)
plt.plot(spectrum_sparse)


# %%
