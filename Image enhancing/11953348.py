import numpy as np
import imageio.v3 as imageio
#Joao Augusto Fernandes Barbosa - 11953348 - SCC0651/2024: Engcomp
    
def root_mean_squared_error(matrix_1 : np.ndarray, matrix_2 : np.ndarray) -> float:
    """Method to compute the difference between two matrix

    Args:
        matrix_1 (np.ndarray): first matrix/image
        matrix_2 (np.ndarray): second matrix/image

    Returns:
        float: Actual value of the root mean squared error
    """
    assert matrix_1.shape == matrix_2.shape, "Erro no processamento. As matrizes nao possuem as mesmas dimensoes"
    temp_matrix = matrix_2 - matrix_1
    temp_matrix = temp_matrix ** 2
    mean_squared = np.mean(temp_matrix)
    root_mean_squared = np.sqrt(mean_squared)
    return root_mean_squared
    
def histogram(matrix : np.ndarray, gray_level : int = 256) -> np.ndarray:
    """Create an 1d array as the histogram for each graylevel present

    Args:
        matrix (np.ndarray): matrix to be analyzed
        gray_level (int, optional): gray level, used in this study as 8bit. Defaults to 256.

    Returns:
        np.ndarray: histogram with X values in range 0 - 255 and Y values as the count of each ocurrence for those pixels
    """
    hist = np.array([np.sum(matrix == i) for i in range(gray_level)])
    return hist

def histogram_equalization(matrix : np.ndarray, hist : np.ndarray, gray_level : int = 256) -> np.ndarray:
    """Function to get a better distribution of an histogram

    Args:
        matrix (np.ndarray): array to be equalized
        gray_level (int, optional): gray level, used in this study as 8bit. Defaults to 256.

    Returns:
        np.ndarray: Matrix after the substitution using the histogram equalization method
    """
    hist_cumulative = hist.cumsum()
    n, m = matrix.shape
    hist_transf = [((gray_level - 1)/float(m * n)) * hist_cumulative[i] for i in range(gray_level)]
    final_matrix = np.zeros([n, m]).astype(np.uint8)
    
    for z in range(gray_level):
        final_matrix[np.where(matrix == z)] = hist_transf[z]
    return final_matrix

def get_super_resolution_image(processed_images : np.ndarray, image_high : np.ndarray) -> np.ndarray:
    """Create an image 2 times bigger using 4 matrix of the same shape

    Args:
        processed_images (np.ndarray): list of images for this method
        image_high (np.ndarray): image objective, to assert the shape

    Returns:
        np.ndarray: final image after the resolution method
    """
    super_resolution_image = np.zeros(image_high.shape)
    num_colunas = image_high.shape[1]
    num_linhas = image_high.shape[0]
    for i in range(num_colunas):
        if i % 2 == 0:
            for j in range(num_linhas):
                if j % 2 == 0:
                    super_resolution_image[j, i] = processed_images[0][j//2][i//2]
                else:
                    super_resolution_image[j, i] = processed_images[1][(j-1)//2][i//2]
        else:
             for j in range(num_linhas):
                if j % 2 == 0:
                    super_resolution_image[j, i] = processed_images[2][j//2][(i-1)//2]
                else:
                    super_resolution_image[j, i] = processed_images[3][(j-1)//2][(i-1)//2]
    return super_resolution_image
                    
def single_image_cumulative_histogram(images_low : np.ndarray, image_high : np.ndarray) -> np.ndarray:
    """First method of image processing using an individual histogram for each image

    Args:
        images_low (np.ndarray): list of images/matrix with low quality
        image_high (np.ndarray): image/matrix with high quality

    Returns:
        np.ndarray: final image after processing
    """
    processed_images = []
    for image in images_low:
        hist = histogram(matrix = image)
        result_image = histogram_equalization(image, hist = hist)
        processed_images.append(result_image)
    return get_super_resolution_image(processed_images, image_high) 

def joint_cumulative_histogram(images_low : np.ndarray, image_high : np.ndarray) -> np.ndarray:
    """A histogrram equalization that uses a matrix calculated as the mean of each one of the images low

    Args:
        images_low (np.ndarray): list with the low quality images
        image_high (np.ndarray): high quality image

    Returns:
        np.ndarray: final image after processing the  superposition
    """
    hist = np.zeros(images_low[0].shape[0])
    processed_images = []
    for image in images_low:
        hist = np.add(hist, histogram(matrix = image))
    hist = (hist//len(images_low))
    for image in images_low:
        result_image = histogram_equalization(image, hist = hist)
        processed_images.append(result_image)
    return get_super_resolution_image(processed_images = processed_images, image_high = image_high) 

def gama_correction(images_low : np.ndarray, image_high : np.ndarray) -> np.ndarray :
    """Image processing using the gama correction. In this case, the function needs a new input by the user, which is the gama factor, stored as a float.

    Args:
        images_low (np.ndarray): list with low quality images
        image_high (np.ndarray): high quality image

    Returns:
        np.ndarray: final image after processing the superposition 
    """
    gama = float(input().strip())
    processed_images = []
    for image  in images_low:
        result_image = ((image/255.0)**(1/gama))*255 
        processed_images.append(result_image)
    return get_super_resolution_image(processed_images=processed_images, image_high= image_high)



def main():
    filename_low = input().strip()
    filename_high = input().strip()
    images_low= [imageio.imread(f'{filename_low}{img_id}.png') for img_id in range(4)]
    image_high = imageio.imread(filename_high)
    preprocessing_method = input().strip()
    match preprocessing_method:
        case '0':
            output = get_super_resolution_image(processed_images=images_low, image_high= image_high)
        case '1':
            output = single_image_cumulative_histogram(images_low, image_high)
        case '2':
            output = joint_cumulative_histogram(images_low= images_low, image_high=image_high)
        case '3':
            output = gama_correction(images_low = images_low, image_high= image_high)
    print(root_mean_squared_error(output, image_high))
main()

