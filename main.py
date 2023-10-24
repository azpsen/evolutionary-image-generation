from PIL import Image
import numpy as np


def fitness(img, target, img_size):
    """
    Get the similarity between two binary image arrays
    :param img: First image to compare
    :param target: Goal image
    :param img_size: Total size of image (width * height)
    :return: Percentage similarity between the two images (0.0 to 1.0)
    """

    return np.sum(np.bitwise_xor(img, target)) / img_size


def evolve(target, target_fitness=1.0, print_interval=-1):
    """
    Starting from an image of noise, evolve until it reaches the desired similarity to the given target
    :param target: Goal image
    :param target_fitness: Similarity to reach before returning
    :param print_interval: How frequently to print fitness and output intermediary image; -1 for no output
    :return: Binary array that is of target_fitness similarity to target
    """

    # Get image size information
    img_width = len(target)
    img_height = len(target[0])
    img_size = img_width * img_height

    # Initialize new image array - random array of 0s and 1s, initial fitness should be ~0.5
    img = np.random.choice(a=[True, False], size=(img_width, img_height))

    # Initialize iter counter and fitness values
    k = 0
    img_fitness = 0.0

    # Keep evolving until target fitness is reached
    while img_fitness < target_fitness:

        # Create a copy of the current best image array
        new_img = np.empty_like(img)
        np.copyto(new_img, img)

        # Choose a pixel to flip at random
        i = np.random.randint(0, img_width)
        j = np.random.randint(0, img_height)
        new_img[i, j] = not new_img[i, j]

        # Calculate fitness of image with random pixel flipped
        new_fitness = fitness(new_img, target, img_size)

        # If new image is better, replace current best with new, otherwise discard
        if new_fitness > img_fitness:
            img = new_img
            img_fitness = new_fitness

        # Output progress image and print fitness on print_interval
        k += 1
        if print_interval >= 0 and k % print_interval == 0:
            print(img_fitness)
            write_image("out.png", img)

    return img


def read_image(path):
    """
    Read an image from a path, convert it to monochrome, and return a binary array of pixels corresponding to the image
    :param path: Path of image to be read
    :return: Binary array of pixels corresponding to the pixels of the given image
    """

    # Read image and convert it to black and white (no grey)
    img = Image.open(path).convert("1")
    pixel_map = img.load()

    # Create empty numpy array to store pixel data in
    b = np.zeros((img.width, img.height), dtype=bool)

    # Read image data into array
    for i in range(img.width):
        for j in range(img.height):
            b[i, j] = pixel_map[i, j]

    return b


def write_image(path, img):
    """
    Convert a binary array to image data and write it to the given path
    :param path: Path to write image to
    :param img: Binary array to convert to image
    :return: None
    """
    im = Image.fromarray(img.T)
    im.save(path)


def main():

    # Read in target image
    target = read_image("input.jpg")

    # Write converted (monochrome, no grayscale) image to file
    write_image("target.png", target)

    # Evolve random noise to the desired similarity
    out = evolve(target, target_fitness=0.7, print_interval=5000)

    # Write final output
    write_image("final_out.png", out)


if __name__ == '__main__':
    main()
