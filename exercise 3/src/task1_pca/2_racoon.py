import utils
import matplotlib.pyplot as plt  # to show the image
import pylab  # to show the image

""" Task 1.2: In this script, we apply principal component analysis to a racoon image. 
We need functions defined in utils.py for this script.
"""

# Load and resize the racoon image in grayscale
image = utils.load_resize_image()
plt.imshow(image)
pylab.show()

# Compute Singular Value Decomposition (SVD) using utils.compute_svd()
# reconstruct_data_using_truncated_svd？
U, S, V_t = utils.compute_svd(image)

# Reconstruct images using utils.reconstruct_images
utils.reconstruct_images(U, S, V_t)

# Compute the number of components where energy loss is smaller than 1% using
# utils.compute_num_components_capturing_threshold_energy()
print(utils.compute_num_components_capturing_threshold_energy(S))
