#!/usr/bin/python3

import argparse
import sys
import os
from tqdm import tqdm
import numpy as np

from skimage.metrics import mean_squared_error as mse, peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
from numpy import asarray
import math
from statistics import mean

def compare(gt,est,resize=False):
    result=dict()
    raw_gt = Image.open(gt).convert('RGB')
    raw_est = Image.open(est).convert('RGB')
    # raw_gt_lab = Image.open(gt).convert('LAB')
    # raw_est_lab = Image.open(est).convert('LAB')
    if resize:
        size = (400,400)
        raw_gt = raw_gt.resize(size)
        raw_est = raw_est.resize(size)
    gt_img = asarray(raw_gt)
    est_img = asarray(raw_est)
    # gt_img_lab = asarray(raw_gt_lab)
    # est_img_lab = asarray(raw_est_lab)
    

    #print(f"gt_img.shape: {gt_img.shape} est_img.shape: {est_img.shape}") 
    
   
    result["psnr"] = psnr(gt_img, est_img,data_range=255) 
    result["rmse"] = math.sqrt(mse(gt_img, est_img))
    # result["rmse_lab"] = math.sqrt(mse(gt_img_lab, est_img_lab))
    result["ssim"] = ssim(gt_img,est_img,channel_axis=2)
    
    return result

def evaluate(gt_dir,est_dir,img_suf=".png",resize=False):
    gt_list = [os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.endswith(img_suf)]
    est_set = set([os.path.splitext(f)[0] for f in os.listdir(est_dir) if f.endswith(img_suf)])
    metrics_list={}
    for idx, img_name in tqdm(enumerate(gt_list)):
        if img_name in est_set:
            gt = os.path.join(gt_dir,img_name+img_suf)
            est = os.path.join(est_dir,img_name+img_suf)
            metrics = compare(gt,est,resize)
            for m in metrics:
                if m not in metrics_list.keys():
                    metrics_list[m]=[]
                metrics_list[m].append(metrics[m]) 
        else:
            print(f"Error: image {img_name} is not generated")
    for metric in metrics_list:
        print(f"{metric}: {mean(metrics_list[metric])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, help='ground truth dir')
    parser.add_argument('--est_dir', type=str, help='generated images dir')
    parser.add_argument('--img_suf', type=str, default='.png', help='Image type')
    parser.add_argument('--resize', default=False, action='store_true')
    parser.add_argument('--no-resize', dest='resize', action='store_false')
    opt = parser.parse_args()
    evaluate(opt.gt_dir, opt.est_dir, opt.img_suf, opt.resize)