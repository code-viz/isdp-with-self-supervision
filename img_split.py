import numpy as np
import glob
import cv2
import os

if __name__ == "__main__":
    
    save_path   = 'data/img/'
    color_path  = 'gray_img/'
    txt_path    = 'data/txt/'
    img_width = 32
    path_project_img = save_path + color_path + txt_path

    for project_img_file in glob.glob(path_project_img + '/*'):
        for class_name in ['/clean/', '/buggy/']:
            for project_clean_img in glob.glob(project_img_file + class_name + '*.png'):
                image = cv2.imread(project_clean_img, cv2.IMREAD_GRAYSCALE)
                height, width = image.shape

                image_num = int(height / img_width) + 1
                pad_size = (image_num * width) - height
                pad = np.zeros((pad_size, img_width))

                full_image = np.concatenate((image, pad), axis=0)

                img_name = os.path.split(project_clean_img)[-1].split('.png')[0]
                path_save = project_img_file + '/split_' + class_name[1:] + img_name

                if not os.path.exists(path_save):
                    os.makedirs(path_save)

                for split in range(image_num):
                    split_image = full_image[split*img_width:(split+1)*img_width,:]
                    cv2.imwrite(path_save + f'/{split}.png', split_image)