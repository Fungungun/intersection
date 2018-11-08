'''
 Created by 
 Chenghuan Liu (Github: Fungungun) 
 3/11/18
'''
import mxnet as mx
from gluoncv import model_zoo, data
import cv2
import numpy as np


class Detector():
    def __init__(self, model='yolo3_darknet53_coco', threshold=0.8, cls_cate='coco', ctx=mx.gpu(0), resize_short=512,
                 cls_of_interest={'coco': [0, 2, 5, 7], 'voc': [5, 6, 14]}):
        '''
        For choices of model check the official site of gluoncv
        The default 'cls of interest' includes person and vehicle
        Default 'COCO' class category

        Some choices:
        Model                             Speed         Accuracy
        faster_rcnn_resnet101_v1d_coco    Slow          (Negative Correlated
        yolo3_darknet53_coco              Fast          With
        ssd_512_mobilenet1.0_coco         Fastest       Speed)
        '''
        self.net = model_zoo.get_model(model, pretrained=True, ctx=ctx)
        self.net.hybridize(static_alloc=True, static_shape=True)  # convert to static graph
        self.threshold = threshold
        self.cls_cate = cls_cate
        self.ctx = ctx
        self.resize_short = resize_short
        self.cls_of_interest = cls_of_interest

    def process(self, frame):
        '''
        :param frame: BGR image from cv2.imread
        :return: List of object classes, list of bboxes [x1,y1,x2,y2,cls,score] and resized BGR image
        '''

        v_height, v_width, _ = frame.shape
        if v_height > v_width:
            show_img = cv2.resize(frame, (self.resize_short, int(1.0 * v_height * self.resize_short / v_width)))
        else:
            show_img = cv2.resize(frame, (int(1.0 * v_width * self.resize_short / v_height), self.resize_short))
        np_img = show_img.copy()

        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        nd_img = mx.nd.array(np_img)

        nd_img, _ = data.transforms.presets.yolo.transform_test(nd_img, short=self.resize_short)

        x = nd_img.as_in_context(self.ctx)
        class_IDs, scores, bounding_boxs = self.net(x)

        class_IDs, scores, bounding_boxs = \
            class_IDs.asnumpy(), scores.asnumpy(), bounding_boxs.asnumpy()

        score = scores[0, :, :].squeeze()
        class_id = class_IDs[0, :, :].squeeze()
        bounding_box = bounding_boxs[0, :, :].squeeze().astype(int)

        index_1 = score > self.threshold
        index_2 = np.array([True if x in self.cls_of_interest[self.cls_cate] else False for x in class_id])
        index = np.logical_and(index_1, index_2)

        class_id = class_id[index]
        bounding_box = bounding_box[index, :]

        return class_id, bounding_box, show_img
