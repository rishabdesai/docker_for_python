"""
ref : https://www.youtube.com/watch?v=8uaDoMuDK6E&list=PLZsOBAyNTZwazmH7F9pAE_fFsKmrqYqq5&index=4
https://scikit-image.org/docs/stable/auto_examples/filters/plot_nonlocal_means.html

"""
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import data, img_as_float, img_as_ubyte
from skimage import io
import numpy as np


image = img_as_float(io.imread("monalisa_noisy.jpg")).astype(np.float32)

# estimate the noise standard deviation from the noisy image
sigma_est = np.mean(estimate_sigma(image, channel_axis=-1))
print(f'estimated noise standard deviation = {sigma_est}')

patch_kw = dict(patch_size=10,      # 5x5 patches
                patch_distance=3,  # 13x13 search area
                channel_axis=-1)

# slow algorithm
denoise_img = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=False,
                               **patch_kw)

denoise_img_as_8byte = img_as_ubyte(denoise_img)

io.imsave("NLM.jpg", denoise_img_as_8byte)
