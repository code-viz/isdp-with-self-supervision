import os
import glob
import numpy
import pandas as pd
from colorMap import *
import cv2



if __name__ == "__main__":

    save_path   = 'data/img/'
    color_path  = 'gray_img/'
    txt_path    = 'data/txt/'
    img_width = 32

    projects = {}
    for txtfile in glob.glob(txt_path + '*.csv'):

        # if txtfile == 'data/txt/ant-1.5.csv':
        #     continue
        # if txtfile == 'data/txt/ant-1.6.csv':
        #     continue
        # if txtfile == 'data/txt/ant-1.7.csv':
        #     continue
        # if txtfile == 'data/txt/poi-3.0.csv':
        #     continue
        
        # data/img/ + gray_img/ + project name
        path_project = save_path + color_path + txtfile.replace(txt_path,'').split('.csv')[0]
        
        if not os.path.exists(path_project):
            os.makedirs(path_project)

        if not os.path.exists(path_project + '/buggy/'):
            os.makedirs(path_project + '/buggy/')
        
        if not os.path.exists(path_project + '/clean/'):
            os.makedirs(path_project + '/clean/')

        filename    = 'data/txt/'+txtfile
        file_df     = pd.read_csv(txtfile)
        df_sample   = file_df[['name', 'version', 'name.1', 'bug']]
        
        name    = df_sample['name'].values.tolist()
        version = df_sample['version'].values.tolist()
        path    = df_sample['name.1'].values.tolist()
        bug     = df_sample['bug'].values.tolist()

        clean_count = 0
        bug_count   = 0
        for name, version, path, bug in zip(name, version, path, bug):
            path        = path.replace('.', '/')
            version     = str(version)

            file_path   = 'data/archives/' + name + '-' + version + '/' + path
            label = 1 if bug > 0 else str(0)

            print(file_path)
            if os.path.exists(file_path + '.java'):
                size = getFileSize(file_path + '.java')
                if size == 0:
                    break
                
                im = getNewColorImg(file_path+'.java', img_width)
                print(im)

                if label == 1:
                    path_save = path_project + '/buggy/' + path.replace('/','_') + '.png'
                    cv2.imwrite(path_save, im)

                    bug_count +=1
                else:
                    path_save = path_project + '/clean/' + path.replace('/','_') + '.png'
                    cv2.imwrite(path_save, im)

                    clean_count += 1
        
        projects[txtfile] = {'clean':clean_count, 'bug':bug_count}
    
    print(projects.items())


                
                
                

