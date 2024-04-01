import numpy as np
import torch.nn as nn
import torch
from math import sqrt

def compute_iou(box_a,box_b):
    x1=np.max(box_a[0],box_b[0])
    x2=np.min(box_a[2],box_b[2])
    y1 = np.max(box_a[1], box_b[1])
    y2 = np.min(box_a[3], box_b[3])

    iner=np.max([x2-x1+1,0])*np.max([y2-y1+1,0])
    outer=(box_a[2]-box_a[0]+1)*(box_a[3]-box_a[1]+1)+(box_b[2]-box_b[0]+1)*(box_b[3]-box_b[1]+1)-iner

    iou=iner/outer
    return iou
