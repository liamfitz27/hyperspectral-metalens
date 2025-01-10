#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning, SparseCoder
from PIL import Image
import os

# LOAD IMAGES
directory = r"forest_images"
dir_list = os.listdir(directory)[:]
valid_idx = np.random.randint(0, len(dir_list), 100)
train_list = np.delete(dir_list, valid_idx)
valid_list = [dir_list[i] for i in valid_idx]
imgs_t = []
imgs_v = []
shape = [75,75]
for file in train_list:
    fname = os.fsdecode(file)
    if fname[-4:] == ".jpg":
        img = Image.open(directory+"/"+fname)
        img = img.resize(shape)
        img = np.asarray(img)
        imgs_t.append(np.ndarray.flatten(img))
for file in valid_list:
    fname = os.fsdecode(file)
    if fname[-4:] == ".jpg":
        img = Image.open(directory+"/"+fname)
        img = img.resize(shape)
        img = np.asarray(img)
        imgs_v.append(np.ndarray.flatten(img))

#%%
# DICTIONARY LEARNING
dict_learner = DictionaryLearning(
    n_components=500, transform_alpha=0.01, fit_algorithm="cd", transform_algorithm="lasso_cd",
    random_state=42, verbose=True, max_iter=100, transform_max_iter=10000, positive_dict=True, positive_code=True,
)

sparse_coeffs = dict_learner.fit(imgs_t).transform(imgs_t)
sparse_dict = dict_learner.components_
sparse_reps = sparse_coeffs@sparse_dict

filepath = directory + r"/forest_images_dictlearning.npz"
np.savez(filepath, sparse_coeffs=sparse_coeffs, sparse_dict=sparse_dict)

#%%
# LOAD/SHOW DICTIONARY REPRESENTATIONS
filepath = directory + r"/forest_images_dictlearning.npz"
dict_learned = np.load(filepath)

sparse_coeffs = dict_learned["sparse_coeffs"]
sparse_dict = dict_learned["sparse_dict"]
sparse_reps = sparse_coeffs @ sparse_dict

i = np.random.randint(len(imgs_t))
img0 = np.reshape(imgs_t[i], (*shape,3))
img1 = np.reshape((sparse_reps[i,:]/np.max(sparse_reps[i,:])*255).astype(np.uint8), (*shape,3))
fig, ax = plt.subplots(1, 2, dpi=200)
ax[0].imshow(img0)
ax[1].imshow(img1)

#%%
# SHOW DICTIONARY BASIS ELEMENTS
i = np.random.randint(np.shape(sparse_dict)[0])
img = np.reshape((sparse_dict[i,:]/np.max(sparse_dict[i,:])*255).astype(np.uint8), (*shape,3))
plt.imshow(img)

#%%
# CODE VALIDATION IMAGES WITH LIBRARY
coder = SparseCoder(
    dictionary=sparse_dict , transform_algorithm='lasso_cd',
    transform_alpha=0.01, positive_code=True,
)

test_coeffs = coder.transform(imgs_v)
test_reps = test_coeffs@sparse_dict
#%%
# SHOW VALIDATION REPRESENTATIONS
i = np.random.randint(np.shape(test_reps)[0])
img0 = np.reshape(imgs_v[i], (*shape,3))
img1 = np.reshape((test_reps[i,:]/np.max(test_reps[i,:])*255).astype(np.uint8), (*shape,3))
fig, ax = plt.subplots(1, 2, dpi=200)
ax[0].imshow(img0)
ax[1].imshow(img1)
#%%
# SAVE RESULTS
np.savez("eigencat_basis", sparse_dict=sparse_dict, sparse_coeffs=sparse_coeffs, test_coeffs=test_coeffs)

#%%
# PLOT FINAL RESULTS
fig, ax = plt.subplots(2, 3, dpi=300)
idx = np.random.randint(0, len(imgs_v), 3)
idx = [193, 431,  29]
for i in range(3):
    img = np.reshape(imgs_v[idx[i]], (*shape,3))
    ax[0,i].imshow(img)
    ax[0,i].set_axis_off()
ax[0,1].set_title("Original")
for i in range(3):
    img_rep = np.reshape(test_reps[idx[i]], (*shape,3))
    ax[1,i].imshow((img_rep/np.max(img_rep) * 255).astype(np.uint8))
    ax[1,i].set_axis_off()
ax[1,1].set_title("Reconstructed")      
fig.tight_layout()

#%%
# fig, ax = plt.subplots(4,3, dpi=300)
# idx = np.random.randint(0, np.shape(sparse_dict)[0], 12)
# for i in range(4):
#     for j in range(3):
#         img = np.reshape(sparse_dict[i*4+j,:], (*shape, 3))
#         ax[i,j].imshow((img/np.max(img) * 255).astype(np.uint8))
#         ax[i,j].set_axis_off()
# ax[0,1].set_title("Learned Dictionary Basis Elements")
# plt.subplots_adjust(left = 0, top = 0.9, right = 0, bottom = 0.1, hspace = 0.5, wspace = 0)
# fig.tight_layout()

from mpl_toolkits.axes_grid1 import ImageGrid

basis_plot = [sparse_dict[i,:] for i in range(16)]
fig = plt.figure(figsize=(4., 4.), dpi=300)
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 4),  # creates 2x2 grid of Axes
                 axes_pad=0.05,  # pad between Axes in inch.
                 )

for ax, im in zip(grid, basis_plot):
    # Iterating over the grid returns the Axes.
    im = np.reshape(im, (*shape, 3))
    ax.imshow((im/np.max(im)* 255).astype(np.uint8))
    ax.set_axis_off()

fig.set_title("Learned Basis Elements")