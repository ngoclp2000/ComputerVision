import cv2 
import numpy as np

# Load the image
img = cv2.imread('./images/Chessboard_0481.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to the frequency domain
dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift the zero-frequency component to the center
dft_shift = np.fft.fftshift(dft)

# Get the magnitude spectrum of the image
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Define local maxium
def local_maximum(spec,win_size,thresh):
    wid, hei = spec.shape
    wid_loop_times = int(wid/win_size)
    hei_loop_times = int(hei/win_size)
    global_max = np.max(spec)
    max_pos = []
    for i in range(wid_loop_times):
        for j in range(hei_loop_times):
            window = spec[win_size*i:win_size*(i+1), win_size*j:win_size*(j+1)]
            local_max = np.max(window)
            if (local_max > (global_max*thresh) and local_max < global_max):
                ind = np.unravel_index(np.argmax(window, axis=None), window.shape)
                max_pos.append((5*i+ind[0], 5*j+ind[1]))
    ### YOUR CODE HERE
    return max_pos

max_pos = local_maximum(magnitude_spectrum, 5, 0.9)

for pos in max_pos:
    # remove sin noise
    dft_shift[pos] = 0
    # for showing purpose only
    magnitude_spectrum[pos] = np.min(magnitude_spectrum)

back_ishift = np.fft.ifftshift(dft_shift)

# Inverse the DFT to get the filtered image
filtered_img = cv2.idft(back_ishift)
filtered_img = cv2.magnitude(filtered_img[:, :], filtered_img[:, :])

# Show the original and filtered images
cv2.imshow('Original', gray)
cv2.imshow('Filtered', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
