import numpy as np
from utils.ps_lib import *
import matplotlib.pyplot as plt
from numpy.fft import fft, ifftn,ifft
from PIL import Image
import imageio
from perlin_noise import PerlinNoise
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter
import sys
from scipy.optimize import minimize, lsq_linear, linprog
from scipy import sparse

def get_noise(dim):
    noise = PerlinNoise(octaves=10, seed=1)
    xpix, ypix = dim[1], dim[0]
    pic = np.array([[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)])
    return pic

def prep_img(img):
    # low-pass filter the img to  avoid messy artifacts in final moire
    img = gaussian_filter(img, sigma=0.1, mode='nearest')

    # Scale dynamic range of image to correspond to domain of the inverse of
    # moire intensity profile
    img = ((img - np.min(img))/4) + (1/8)
    return img

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(loc=0,scale=sigma,
                             size = img.shape)
    return img + noise

def get_luminance(img):
    Y = img[:,:,0]*0.3 + img[:,:,1]*0.6 + img[:,:,2]*0.1
    return Y

#===============================================================================

def make_gratings(target_img, name, add_noise=False):

    target_img = prep_img(target_img)

    plt.imshow(target_img, cmap='gray')
    plt.show()

    dim = target_img.shape
    xcors, ycors  = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))

    p1 = np.vectorize(lambda x: 0.5 + 0.5*np.cos(2*np.pi*x))
    p2 = p1
    phi_1 = np.vectorize(lambda x,y: 0.5*np.pi *x)
    # phi_1 = np.vectorize(lambda x,y: np.pi*2*0.25*(x*np.cos(np.pi/6) + y*np.sin(np.pi/6)))
    #phi_1 = np.vectorize(lambda x,y: 0.5*np.pi *x**2 +  0.5*np.pi *y**2 )

    phase_img = phi_1(xcors,ycors)

    # inv_p(): range (0,1/2), domain(1/8, 3/8)
    # phase modulation is equal to inv_p(target_img)
    phase_modulation = (np.arccos(-2+8*target_img))/(2*np.pi)

    noise = 0
    if add_noise:
        noise = get_noise(dim)

    L2 = p2(phase_img + phase_modulation/2 + noise)
    L1 = p1(phase_img - phase_modulation/2 + noise)


    plt.imshow(L1, cmap='gray')
    plt.show()
    plt.imshow(L2, cmap='gray')
    plt.show()

    plt.imshow(L1*L2, cmap='gray')
    plt.show()

    imageio.imwrite(f"{name}_grating1.jpg",L1)
    imageio.imwrite(f"{name}_grating2.jpg",L2)
    imageio.imwrite(f"{name}_sup.jpg", L1*L2)

#===============================================================================

def make_art_gratings(img1,img2, name):
    img1 = resize(img1, (512,512))
    img2 = resize(img2 , (512, 512))
    img1 = (img1 - np.min(img1))/(np.max(img1) - np.min(img1))
    img2 = (img2 - np.min(img2))/(np.max(img2) - np.min(img2))

    img1 = prep_img(img1)
    img2 = prep_img(img2)


    plt.imshow(img1, cmap='gray')
    plt.show()
    plt.imshow(img2, cmap='gray')
    plt.show()


    shift = int(0.2 * 512)
    xcors, ycors  = np.meshgrid(np.arange(512), np.arange(512))
    xcors_extra, ycors_extra  = np.meshgrid(np.arange(512), np.arange(512  + shift))

    p1 = np.vectorize(lambda x: 0.5 + 0.5*np.cos(2*np.pi*x))
    p2 = p1


    phase_mod1 = (np.arccos(-2+8*img1))/(2*np.pi)
    #phase_mod1 *= 1/np.mean(img1)

    phase_mod2 = (np.arccos(-2+8*img2))/(2*np.pi)
    #phase_mod2 *= 1/np.mean(img2)


    # Encoding objecting function in matrix A
    top = sparse.hstack((sparse.eye(512**2), -1*sparse.eye(512**2), sparse.csr_matrix((512**2, shift*512))))
    bottom = sparse.hstack((sparse.eye(512**2),sparse.csr_matrix((512**2, shift*512)),-1*sparse.eye(512**2)))
    A = sparse.vstack((top, bottom))


    # Encoding second derivative smoothness constraint for phi1
    data = np.ones((3,512**2))
    data[:,1] = -2
    offsets = [512,0, -512]
    finite_difference_phi1 = sparse.dia_matrix((data, offsets), shape=(512**2,512**2))
    finite_difference_phi1 = sparse.hstack((finite_difference_phi1,
                                            sparse.csr_matrix((512**2,
                                                               512*(512+shift)))))


    # Encoding second derivative smoothness constraint for phi2
    data = np.ones((3,512*(512+shift)))
    data[:,1] = -2
    offsets = [512,0, -512]
    finite_difference_phi2 = sparse.dia_matrix((data, offsets), shape=(512*(512+shift),
                                                                       512*(512+shift)))
    finite_difference_phi2 = sparse.hstack((sparse.csr_matrix((512*(512+shift),
                                                               512**2)),
                                             finite_difference_phi2))


    # Combining smoothness constraint for phi1 and phi2
    finite_difference = sparse.vstack((finite_difference_phi1,
                                       finite_difference_phi2))

    # Adding smoothness constraint to objective
    A = sparse.vstack((A, finite_difference))


    # Making the result vector, RHS of linear equation
    result = np.vstack((phase_mod1.reshape((512**2, 1)),
                        phase_mod2.reshape((512**2, 1)),
                        np.zeros((512**2 + 512*(512+shift), 1))))
    result = np.ndarray.flatten(result)


    sol = sparse.linalg.lsqr(sparse.linalg.aslinearoperator(A), result)


    phase_img_1 = (sol[0][:512**2]).reshape((512,512))
    phase_img_2 = (sol[0][512**2:]).reshape((512+shift,512))

    carrier_phase = np.vectorize(lambda x,y: 0.5*np.pi *x)

    # Make the gratings
    L1 = p1(phase_img_1 + carrier_phase(xcors,ycors))
    L2 = p2(phase_img_2  + carrier_phase(xcors_extra, ycors_extra))

    plt.imshow(L1, cmap='gray')
    plt.show()
    plt.imshow(L2, cmap='gray')
    plt.show()

    sup1 = L2 * np.pad(L1, ((0, shift), (0,0)), constant_values=1)
    sup2 = L2 * np.pad(L1, ((shift,0), (0,0)), constant_values=1 )

    plt.imshow(sup1, cmap='gray')
    plt.show()
    plt.imshow(sup2, cmap='gray')
    plt.show()

    imageio.imwrite(f"{name}_grating1.jpg",L1)
    imageio.imwrite(f"{name}_grating2.jpg",L2)
    imageio.imwrite(f"{name}_sup1.jpg", sup1)
    imageio.imwrite(f"{name}_sup2.jpg", sup2)

#===============================================================================

if __name__ == "__main__":
    type = input("Type 'art' or 'simple': \n")
    type = type.strip()
    if type == 'simple':
        img_path = input("Enter a target image file path:\n")
        img_path = img_path.strip()

        name = input("Enter a name (without the file extension) for the resulting files, \
                <name>_grating1.jpg, <name>_grating2.jpg, <name>_sup.jpg:\n")
        name = name.strip()

        noise = input("Add noise? Type Y or N:\n")
        noise = noise.strip()

        add_noise = False
        if noise == "Y":
            add_noise = True

        img = read_image(img_path)
        img = (img - np.min(img))/(np.max(img) - np.min(img))
        #img = img - gaussian_filter(img, sigma=1, mode='nearest')
        #imageio.imwrite(f"{name}_only_hi_freq.jpg",img)
        make_gratings(get_luminance(img), name, add_noise)
    if type == 'art':
        img_path1 = input("Enter a target image 1 file path:\n")
        img_path1 = img_path1.strip()

        img_path2 = input("Enter a target image 2 file path:\n")
        img_path2 = img_path2.strip()

        name = input("Enter a name (without the file extension) for the resulting files, \
                <name>_grating1.jpg, <name>_grating2.jpg, <name>_sup1.jpg, <name>_sup2.jpg:\n")
        name = name.strip()


        img1 = read_image(img_path1)
        img1 = (img1 - np.min(img1))/(np.max(img1) - np.min(img1))
        img2 = read_image(img_path2)
        img2 = (img2 - np.min(img2))/(np.max(img2) - np.min(img2))

        make_art_gratings(get_luminance(img1), get_luminance(img2), name)
