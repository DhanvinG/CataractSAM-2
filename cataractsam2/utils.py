import numpy as np, matplotlib.pyplot as plt

def show_mask(mask, ax, obj_id=0):
    cmap  = plt.get_cmap("tab10")
    color = np.array([*cmap(obj_id)[:3], 0.6])
    h, w  = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))

def show_points(coords, labels, ax, ms=200):
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    ax.scatter(pos[:,0], pos[:,1], c="lime", marker="*", s=ms, edgecolor="w")
    ax.scatter(neg[:,0], neg[:,1], c="red",  marker="*", s=ms, edgecolor="w")
