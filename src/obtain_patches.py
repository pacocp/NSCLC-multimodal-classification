"""Script to extract and save patches."""
from cv2 import imread, imwrite
from pathlib import Path
from tqdm import tqdm
import numpy as np 
import os

save_path = '../all_images_patches/'
images_path = '../all_images_converted'
paths = []
labels = []
log_file = open('log_svs2png.txt', 'w')
patch_size = 512

for filename in tqdm(Path('../all_images_converted/').glob('*.jpeg')):
    f = str(filename)
    save_name = f.split('/')[-1].split('.')[0]
    im = imread(f)
    nobgr_img_blocks = []
    for j in range(0,im.shape[1],patch_size):
        for i in range(0,im.shape[0],patch_size):
            block = im[i:i+patch_size, j:j+patch_size]
            if block.shape == (patch_size, patch_size, 3):
                if(np.mean(block[:][:][0]) < 220.0 and np.mean(block[:][:][1]) < 220.0
                and np.mean(block[:][:][2]) < 220.0):
                    nobgr_img_blocks.append(block)
    index = 0
    for block in nobgr_img_blocks:
        if not os.path.isfile(save_path + save_name + '_' + str(index) + '.jpeg'): 
            try:
                imwrite(save_path + save_name + '_' + str(index) + '.jpeg', block)
            except:
                log_file.write(f+'\n') 
                continue
        else:
            print('already converted')
        
        index += 1
log_file.close()

