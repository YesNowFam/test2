import numpy as np
import PIL.Image

def generate_dataset_preview_grid(training_set, random_seed=0):
    """
    Set up a grid of sample images from the training set for visualization.

    This function creates a grid of images sampled from the training set. If the
    training set has labels, it attempts to create a diverse grid by sampling from
    different label groups.

    Args:
        training_set (object): The training dataset object.
        random_seed (int, optional): Seed for the random number generator. Defaults to 0.

    Returns:
        tuple: A tuple containing three elements:
            - tuple: Grid dimensions (width, height)
            - numpy.ndarray: Stacked array of sampled images
            - numpy.ndarray: Stacked array of corresponding labels

    The grid size is determined based on the image resolution to fit within
    7680x4320 pixels, with a minimum of 4x7 and maximum of 32x32 images.
    """
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)



def save_image_grid(img, fname, drange, grid_size):
    """
    Save a grid of images to a file.

    This function takes a batch of images, arranges them in a grid, and saves
    the resulting image to a file.

    Args:
        img (numpy.ndarray): A batch of images as a 4D array (NCHW format).
        fname (str): The filename to save the image grid to.
        drange (tuple): The dynamic range of the input images as (min, max).
        grid_size (tuple): The dimensions of the grid as (width, height).

    The function performs the following steps:
    1. Normalizes the image data to the range [0, 255].
    2. Reshapes the batch of images into a grid.
    3. Saves the resulting image using PIL, handling both grayscale and RGB images.

    Note: This function assumes that the input images are in NCHW format
    (batch size, channels, height, width) and have either 1 or 3 channels.
    """
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)