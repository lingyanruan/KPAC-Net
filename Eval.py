#! /usr/bin/python
# -*- coding: utf8 -*-
import matplotlib.pyplot as plt 
import os, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config
from skimage import measure,io


ni = 1

def modcrop(imgs, modulo):

    tmpsz = imgs.shape
    sz = tmpsz[0:2]

    h = sz[0] - sz[0]%modulo
    w = sz[1] - sz[1]%modulo
    imgs = imgs[0:h, 0:w,:]
    return imgs

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs


def DefocusDeblur():
    # weight path
    checkpoint_dir = './checkpoints'
    weight_path = checkpoint_dir + '/KPAC-weight.npz' 


    ## create folders to save result images
    save_dir = './Evaluations/single_results_3level'
    tl.files.exists_or_mkdir(save_dir)


    valid_ref_img_list = sorted(tl.files.load_file_list(path=config.TEST.folder_path_c, regx='.*.png', printable=False))
    valid_gt_img_list = sorted(tl.files.load_file_list(path=config.TEST.folder_path_gt, regx='.*.png', printable=False))
    f_psnr = open(save_dir + '_psnr.txt', 'w+')
    f_ssim = open(save_dir + '_ssim.txt', 'w+')


    H = 1120
    W = 1680
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    t_image = tf.placeholder('float32', [1, H, W, 3], name='t_image')
    ###====================== BUILD GRAPH ===========================###  
    with tf.variable_scope('main_net') as scope:
        
        # net_g = Defocus_Deblur_Net6_ms(t_image, ks=5, bs=2, is_train=False, hrg = H, wrg = W) # 2-level   
        net_g = Defocus_Deblur_Net6_ds(t_image, ks=5, bs=2, is_train=False, hrg = H, wrg = W) # 3-level        
    result = net_g.outputs

    tl.files.load_and_assign_npz_dict(name = weight_path, sess = sess)



    ###====================== PRE-LOAD DATA ===========================###        
    valid_ref_imgs = read_all_imgs(valid_ref_img_list, path=config.TEST.folder_path_c, n_threads=10)
    valid_ref_imgs_gt = read_all_imgs(valid_gt_img_list, path=config.TEST.folder_path_gt, n_threads=10)

    tl.files.exists_or_mkdir(save_dir+'/')
    n_iter = 100
    if len(valid_ref_img_list) < 100:
        n_iter = len(valid_ref_img_list) 
    psnr_array = []
    ssim_array = []

    for imid in range(n_iter):
        gt_valid = valid_ref_imgs_gt[imid]/255.0
        
        valid_ref_img = np.expand_dims(valid_ref_imgs[imid],0)     
        valid_ref_img = tl.prepro.threading_data(valid_ref_img, fn=scale_imgs_fn)   # rescale to ［－1, 1]

        ###======================= EVALUATION =============================###
        start_time = time.time()    
        out = sess.run(result, {t_image: valid_ref_img})                
        # print("took: %4.4fs" % ((time.time() - start_time)))

        # print("[*] save images")
        tl.vis.save_image(out[0], save_dir+'/' + valid_ref_img_list[imid][0:-4] + '.png')
        # print('size',out[0].shape)

        img= (io.imread(save_dir+'/' + valid_ref_img_list[imid][0:-4] + '.png' )/255.).astype(np.float32)
        

        psnr_score = measure.compare_psnr(img,gt_valid)
        ssim_score = measure.compare_ssim(img,gt_valid, multichannel=True,data_range =1.0)

        psnr_array.append(psnr_score)
        ssim_array.append(ssim_score)
        f_psnr.write(str(psnr_score)+'\n')
        f_ssim.write(str(ssim_score)+'\n')

    print('************mean value**********************',np.mean(psnr_array),np.mean(ssim_array))

    f_psnr.write('MEAN_PSNR:' + str(np.mean(psnr_array))+'\n')
    f_ssim.write('MEAN_SSIM:' + str(np.mean(ssim_array))+'\n')
    f_psnr.close()
    f_ssim.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='evaluate', help='train, evaluate')    
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    DefocusDeblur()
