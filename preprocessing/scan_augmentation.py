import numpy as np
import SimpleITK as sitk
import random
import math


def rotate_ct_scan(ct_array, angle_range=(0, 360)):
    """
    Rotates the 3D CT scan by a random angle along one of the three axes.

    Parameters:
    - ct_array: 3D NumPy array representing the CT scan
    - angle_range: Tuple (min_angle, max_angle) to define the range of rotation angles

    Returns:
    - rotated_ct: 3D NumPy array after rotation
    """
    # Generate a random angle
    angle = random.uniform(angle_range[0], angle_range[1])

    # Convert to SimpleITK Image
    sitk_image = sitk.GetImageFromArray(ct_array)

    # Choose a random axis of rotation (X, Y, or Z)
    axis = random.choice(['x', 'y', 'z'])

    # Create rotation matrix based on the chosen axis
    if axis == 'x':
        rotation = sitk.Euler3DTransform()
        rotation.SetRotation(angle, 0, 0)
    elif axis == 'y':
        rotation = sitk.Euler3DTransform()
        rotation.SetRotation(0, angle, 0)
    elif axis == 'z':
        rotation = sitk.Euler3DTransform()
        rotation.SetRotation(0, 0, angle)

    # Apply rotation
    rotated_image = sitk.Resample(sitk_image, sitk_image.GetSize(), rotation)

    # Convert back to NumPy array
    rotated_ct = sitk.GetArrayFromImage(rotated_image)
    return rotated_ct


def flip_ct_scan(ct_array):
    """
    Flips the 3D CT scan along a random axis.

    Parameters:
    - ct_array: 3D NumPy array representing the CT scan

    Returns:
    - flipped_ct: 3D NumPy array after flipping
    """
    flip_axis = random.choice([0, 1, 2])  # Choose one of the three axes (x, y, z)

    flipped_ct = np.flip(ct_array, axis=flip_axis)

    return flipped_ct


def swap_ct_scan(ct_array):
    """
    Swaps axes of the 3D CT scan randomly (e.g., swapping XY, XZ, YZ planes) but only within the compatible dimensions.

    Parameters:
    - ct_array: 3D NumPy array representing the CT scan

    Returns:
    - swapped_ct: 3D NumPy array after swapping axes
    """
    shape = ct_array.shape

    if shape[0] == shape[1]:
        swap_order = [1, 0, 2]
    else:
        # If x and y are not the same, restrict to (x, z), (y, z) swaps
        swap_order = random.choice([(0, 2), (1, 2)])

    # Perform the swap
    swapped_ct = ct_array.transpose(swap_order)
    return swapped_ct


def augment_ct_scan(ct_array):
    """
    Performs random rotation, flipping, and axis swapping on the 3D CT scan for data augmentation.

    Parameters:
    - ct_array: 3D NumPy array representing the CT scan

    Returns:
    - augmented_ct: 3D NumPy array after augmentation
    """
    # Apply random rotation
    rotated_ct = rotate_ct_scan(ct_array)

    # Apply random flipping
    flipped_ct = flip_ct_scan(rotated_ct)

    # Apply random axis swapping
    augmented_ct = swap_ct_scan(flipped_ct)

    return augmented_ct

if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt

    img = cv2.imread("../data/jap.png")
    plt.imshow(img)
    plt.show()
    plt.imshow(rotate_ct_scan(img, (0, 90)))
    plt.show()