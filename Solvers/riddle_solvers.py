# Add the necessary imports here
import pandas as pd
import torch
# importing the sys module
import sys         
import numpy as np
import torchvision.transforms as transforms
import cv2
# appending the directory of mod.py 
# in the sys.path list
sys.path.append('D:/3rd-cmp/HackTrick24/SteganoGAN')        

# now we can import mod


from utils import *
from collections import Counter
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg


def solve_cv_easy(test_case: tuple) -> list:
    shredded_image, shred_width = test_case
    shredded_image = np.array(shredded_image)
    # """
    # This function takes a tuple as input and returns a list as output.

    # Parameters:
    # input (tuple): A tuple containing two elements:
    #     - A numpy array representing a shredded image.
    #     - An integer representing the shred width in pixels.

    # Returns:
    # list: A list of integers representing the order of shreds. When combined in this order, it builds the whole image.
    # """
    return []


# def solve_cv_medium(input: tuple) -> list:
#     combined_image_array , patch_image_array = test_case
#     combined_image = np.array(combined_image_array,dtype=np.uint8)
#     patch_image = np.array(patch_image_array,dtype=np.uint8)


#     # """
#     # This function takes a tuple as input and returns a list as output.

#     # Parameters:
#     # input (tuple): A tuple containing two elements:
#     #     - A numpy array representing the RGB base image.
#     #     - A numpy array representing the RGB patch image.

#     # Returns:
#     # list: A list representing the real image.
#     # """
#     return []

def solve_cv_medium(input: tuple) -> list:
    combined_image_array , patch_image_array = input
    combined_image = np.array(combined_image_array,dtype=np.uint8)
    patch_image = np.array(patch_image_array,dtype=np.uint8)
    large_img=combined_image
    patch_img=patch_image
    large_img_gray = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)
    patch_img_gray = cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints_large, descriptors_large = sift.detectAndCompute(large_img_gray, None)
    keypoints_patch, descriptors_patch = sift.detectAndCompute(patch_img_gray, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)    
    matches = flann.knnMatch(descriptors_patch, descriptors_large, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints_patch[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_large[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = patch_img_gray.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        mask = np.zeros(large_img_gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst), 255)
        inpainted_img = cv2.inpaint(large_img, mask, 3, cv2.INPAINT_TELEA)
        inpainted_img=inpainted_img.tolist()
        return inpainted_img
    else:
        print("Not enough matches are found - %d/%d" % (len(good_matches), 4))
        return []


def solve_cv_hard(input: tuple) -> int:
    extracted_question, image = test_case
    image = np.array(image)
    # """
    # This function takes a tuple as input and returns an integer as output.

    # Parameters:
    # input (tuple): A tuple containing two elements:
    #     - A string representing a question about an image.
    #     - An RGB image object loaded using the Pillow library.

    # Returns:
    # int: An integer representing the answer to the question about the image.
    # """
    return 0


def solve_ml_easy(data: pd.DataFrame) -> list:
    df = pd.DataFrame(data)
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)
    df = df[df['visits'] <= 40]
    df = df.resample("1D").mean().fillna(method='ffill')
    cutoff_test = int(len(df) *1 )
    y_train = df.iloc[:cutoff_test]
    model = AutoReg(y_train, lags=10).fit()
    
    return list(model.forecast(50))


def solve_ml_medium(input: list) -> int:
    # """
    # This function takes a list as input and returns an integer as output.

    # Parameters:
    # input (list): A list of signed floats representing the input data.

    # Returns:
    # int: An integer representing the output of the function.
    # """
    return 0



def solve_sec_medium(input: torch.Tensor) -> str:
    img = torch.tensor(img)
    transformed_images = []
    sigmas = [1.0, 2.0, 3.0]  # Example sigmas
    for sigma in sigmas:
        transform = transforms.GaussianBlur(kernel_size=3, sigma=sigma)
        transformed_image = transform(img)
        transformed_images.append(transformed_image)
        
    # """
    # This function takes a torch.Tensor as input and returns a string as output.

    # Parameters:
    # input (torch.Tensor): A torch.Tensor representing the image that has the encoded message.

    # Returns:
    # str: A string representing the decoded message from the image.
    # """
    return ''





def permutedChoiceOne(key64):
    key56_perm = np.array([57, 49, 41, 33, 25, 17, 9,
        1, 58, 50, 42, 34, 26, 18,
        10, 2, 59, 51, 43, 35, 27,
        19, 11, 3, 60, 52, 44, 36,
        63, 55, 47, 39, 31, 23, 15,
        7, 62, 54, 46, 38, 30, 22,
        14, 6, 61, 53, 45, 37, 29,
        21, 13, 5, 28, 20, 12, 4])
    key56=""
    for i in range (0,56):
        key56=key56+key64[key56_perm[i]-1]
    return key56

def shiftLeft(key56 , round):
    key56_c = key56[0:28] 
    key56_d = key56[28:56]
    if round == 1 or round == 2 or round == 9 or round == 16:
        keyLeft = key56_c[1:] + key56_c[0]
        keyRight = key56_d[1:] + key56_d[0]
    else:
        keyLeft = key56_c[2:] + key56_c[:2]
        keyRight = key56_d[2:] + key56_d[:2] 
    return keyLeft+keyRight

def permutedChoiceTwo(keyShifted):
    key48_perm =np.array([14, 17, 11, 24, 1, 5,
            3, 28, 15, 6, 21, 10,
            23, 19, 12, 4, 26, 8,
            16, 7, 27, 20, 13, 2,
            41, 52, 31, 37, 47, 55,
            30, 40, 51, 45, 33, 48,
            44, 49, 39, 56, 34, 53,
            46, 42, 50, 36, 29, 32])
    key48=""
    for i in range (0,48):
        key48=key48+keyShifted[key48_perm[i]-1]
    return key48

def initialPermutation(pt):
    initial_perm =np.array([58, 50, 42, 34, 26, 18, 10, 2,
                60, 52, 44, 36, 28, 20, 12, 4,
                62, 54, 46, 38, 30, 22, 14, 6,
                64, 56, 48, 40, 32, 24, 16, 8,
                57, 49, 41, 33, 25, 17, 9, 1,
                59, 51, 43, 35, 27, 19, 11, 3,
                61, 53, 45, 37, 29, 21, 13, 5,
                63, 55, 47, 39, 31, 23, 15, 7])
    ptPermuted=""
    for i in range (0,64):
        ptPermuted=ptPermuted+pt[initial_perm[i]-1]
    return ptPermuted

def expansion(pt):
    expan = np.array([32, 1, 2, 3, 4, 5, 4, 5,
         6, 7, 8, 9, 8, 9, 10, 11,
         12, 13, 12, 13, 14, 15, 16, 17,
         16, 17, 18, 19, 20, 21, 20, 21,
         22, 23, 24, 25, 24, 25, 26, 27,
         28, 29, 28, 29, 30, 31, 32, 1])
    ptExpanded=""
    for i in range (0,48):
        ptExpanded=ptExpanded+pt[expan[i]-1]
    return ptExpanded

def xor(x,y,fill):
    xorRes = int(x, 2)^int(y, 2)
    xorRes = bin(xorRes)[2:].zfill(fill)
    return xorRes

def permutation(substituted):
    perm = np.array([16,  7, 20, 21,
       29, 12, 28, 17,
       1, 15, 23, 26,
       5, 18, 31, 10,
       2,  8, 24, 14,
       32, 27,  3,  9,
       19, 13, 30,  6,
       22, 11,  4, 25])
    permuted=""
    for i in range (0,32):
        permuted=permuted+substituted[perm[i]-1]
    return permuted

def inversePermutation(pt):
    inverse_perm =np.array([40, 8, 48, 16, 56, 24, 64, 32,
              39, 7, 47, 15, 55, 23, 63, 31,
              38, 6, 46, 14, 54, 22, 62, 30,
              37, 5, 45, 13, 53, 21, 61, 29,
              36, 4, 44, 12, 52, 20, 60, 28,
              35, 3, 43, 11, 51, 19, 59, 27,
              34, 2, 42, 10, 50, 18, 58, 26,
              33, 1, 41, 9, 49, 17, 57, 25])
    ptInverslyPermuted=""
    for i in range (0,64):
        ptInverslyPermuted=ptInverslyPermuted+pt[inverse_perm[i]-1]
    return ptInverslyPermuted

def substitution(xorRes):
    sbox = np.array([[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
         [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
         [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
         [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],
        [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
         [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
         [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
         [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],
        [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
         [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
         [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
         [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],
        [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
         [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
         [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
         [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],
        [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
         [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
         [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
         [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],
        [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
         [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
         [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
         [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],
        [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
         [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
         [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
         [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],
        [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
         [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
         [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
         [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]])
    substituted=""
    for i in range(0,8):
        sub=xorRes[i*6:i*6+6]
        row=int(sub[0]+sub[5], 2)
        col=int(sub[1:5], 2)
        substituted=substituted+bin(sbox[i][row][col])[2:].zfill(4)
    return substituted
def solve_sec_hard(input:tuple)->str:
    """
    This function takes a tuple as input and returns a list a string.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A key 
        - A Plain text.

    Returns:
    list:A string of ciphered text
    """
    key=input[0]
    pt=input[1]
    
    binaryKey=f'{int(key, 16):0>{64}b}'
    key56=permutedChoiceOne(binaryKey)
    binaryPT=f'{int(pt, 16):0>{64}b}'
    permutedPT=initialPermutation(binaryPT)
    ptLeft=permutedPT[0:32]
    ptRight=permutedPT[32:64]

    for i in range(1,17):
        keyShifted=shiftLeft(key56,i)
        keyLeft=keyShifted[0:28] 
        keyRight=keyShifted[28:56]
        keyPermuted=permutedChoiceTwo(keyShifted)
        expandedPTRight=expansion(ptRight)
        xorPTRight=xor(expandedPTRight,keyPermuted,48)
        substitutedPTRight=substitution(xorPTRight)
        permutedPTRight=permutation(substitutedPTRight)
        xorRes=xor(permutedPTRight,ptLeft,32)
        ptLeft=ptRight
        ptRight=xorRes
        key56=keyShifted
    
    inverslyPermutedSwapped=inversePermutation(ptRight+ptLeft)
    return f'{int(inverslyPermutedSwapped, 2):X}'

    
    
    return ''

def solve_problem_solving_easy(input: tuple) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A list of strings representing a question.
        - An integer representing a key.

    Returns:
    list: A list of strings representing the solution to the problem.
    """
    word_freq = Counter(input[0])
    x=input[1]
    sorted_words = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
    return [word for word, freq in sorted_words[:x]]
    


def solve_problem_solving_medium(input: str) -> str:
    """
    This function takes a string as input and returns a string as output.

    Parameters:
    input (str): A string representing the input data.

    Returns:
    str: A string representing the solution to the problem.
    """
    s=input
    stack = []
    res = ""
    i = 0
    while i < len(s):
        if s[i].isdigit():
            count = ""
            while s[i] != '[':
                count += s[i]
                i += 1
            stack.append((int(count) - 1, len(res)))
        elif s[i] == ']':
            count, index = stack.pop()
            temp_str = res[index:]
            for _ in range(count):
                res += temp_str
        else:
            res += s[i]
        i += 1
    return res
    


def traveler (input: tuple,memo):

    i=input[0]
    j=input[1]
    if memo[i][j] != -1:
        return memo[i][j]
    if i == 1 and j == 1:
        return 1
    if i == 0 or j == 0:
        return 0
    memo[i][j] = traveler((i - 1, j),memo) + traveler((i, j - 1),memo)
    return memo[i][j]

def solve_problem_solving_hard(input: tuple) -> int:
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two integers representing m and n.

    Returns:
    int: An integer representing the solution to the problem.
    """
    memo =  np.full((101, 101), -1)
    return int(traveler(input,memo))


riddle_solvers = {
    'cv_easy': solve_cv_easy,
    'cv_medium': solve_cv_medium,
    'cv_hard': solve_cv_hard,
    'ml_easy': solve_ml_easy,
    'ml_medium': solve_ml_medium,
    'sec_medium_stegano': solve_sec_medium,
    'sec_hard':solve_sec_hard,
    'problem_solving_easy': solve_problem_solving_easy,
    'problem_solving_medium': solve_problem_solving_medium,
    'problem_solving_hard': solve_problem_solving_hard
}
