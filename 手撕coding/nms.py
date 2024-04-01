import numpy as np

def nms(dets,iou_thresh,cfd_thred):
    if len(dets)==0:
        return [],[]
    bboxes= np.array(dets)

    x1=bboxes[:,0]
    y1=bboxes[:,1]
    x2=bboxes[:,0]
    y2=bboxes[:,1]

    scores=bboxes[:,4]
    areas=(x2-x1+1)*(y2-y1+1)

    order=np.argsort(scores)
    picked_boxes=[]
    while order.size>0:
        index=order[-1]
        picked_boxes.append(dets[index])

        # 获取当前置信度最大的候选框与其他任意候选框的相交面积
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        # 计算交集部分面积
        intersection = w * h
        # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框删除
        ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ious < iou_thresh)
        order = order[left]
    return picked_boxes


