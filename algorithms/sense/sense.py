import numpy as np



def fft2c(image):
    image = np.fft.ifftshift(image, axes=(-2, -1))
    kspace = np.fft.fft2(image, axes=(-2, -1))
    kspace = np.fft.fftshift(kspace, axes=(-2, -1))
    return kspace


def ifft2c(kspace):
    kspace = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(kspace, axes=(-2, -1))
    image = np.fft.fftshift(image, axes=(-2, -1))
    return image


def sense(coil_images, sens_maps, acceleration):
    """
    SENSE coil combination: Naive implementation
    """
    image_sense = np.full(coil_images.shape[-2:], np.NaN, dtype=np.complex64)

    num_cols = coil_images.shape[-1]
    num_rows = coil_images.shape[-2]
    for col in range(num_cols):  # for each col
        
        # If the column is filled, then skip
        if not np.isnan(np.sum(image_sense[:, col])):
            continue

        superimp_cols = np.array([col + int(i/acceleration * num_cols) for i in range(acceleration)]) # Size (Np,)

        for row in range(num_rows):
            
            S = sens_maps[:, row, superimp_cols]  # Size (Nc x Np)
            U = np.linalg.pinv(S) # Size (Np x Nc)
            a = coil_images[:, row, col] # Size (Nc,)
            v = np.dot(U, a) # Size (Np,)

            image_sense[row, superimp_cols] = v

    return image_sense


def sense_vectorized(coil_images, sens_maps, acceleration):
    """
    SENSE coil combination: Vectorized implementation
    """
    image_sense = np.full(coil_images.shape[-2:], np.NaN, dtype=np.complex64)

    num_cols = coil_images.shape[-1]
    for col in range(num_cols):  # for each col
        
        # If the column is filled, then skip
        if not np.isnan(np.sum(image_sense[:, col])):
            continue

        superimp_cols = np.array([col + int(i/acceleration * num_cols) for i in range(acceleration)]) # Size Np
        
        # Vectorized operations
        S = sens_maps[:, :, superimp_cols]  # Size (Nc x num_rows x Np)
        S = S.transpose(1,0,2) # Size (num_rows x Nc x Np)
        U = np.linalg.pinv(S) # Size (num_rows x Np x Nc)
        a = coil_images[:, :, col] # Size (Nc x num_rows)
        a = np.expand_dims(a, axis=0).transpose(2,1,0) # Size (num_rows x Nc x 1)
        v = np.matmul(U, a) # Size (num_rows x Np x 1)
        image_sense[:, superimp_cols] = np.squeeze(v)

    return image_sense