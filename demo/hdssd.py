import sys
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2


class Head_detection():
    def __init__(self, path):
        self.inference_list = []
        self.count = 0

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

    def draw_boxes(self, image, scores, boxes, classes, im_width, im_height):
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        heads = list()
        idx = 1

        for score, box, name in zip(scores, boxes, classes):
            if name == 1 and score > 0.8: #0.8,0.73
                left = int((box[1])*im_width)
                top = int((box[0])*im_height)
                right = int((box[3])*im_width)
                bottom = int((box[2])*im_height)
                cropped_head = np.array(image[top:bottom, left:right])

                width = right - left
                height = bottom - top
                bottom_mid = (left + int(width / 2), top + height)
                confidence = score
                mydict = {
                    "head_id": idx,
                    "width": width,
                    "height": height,
                    "cropped":cropped_head,
                    "left": left,
                    "right": right,
                    "top": top,
                    "bottom": bottom,
                    "confidence": confidence,
                    "label": None,
                    "bottom_mid": bottom_mid,
                    "model_type": 'FROZEN_GRAPH'
                    }
                heads.append(mydict)
                idx += 1

                # cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2, 8)
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2, 8)
                # cv2.putText(image, 'score: {:.2f}%'.format(score), (left-5, top-5), 0, 0.55, (0,255,255),2)

        return image, heads


    def run(self, image, im_width, im_height):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        self.inference_list.append(elapsed_time)
        self.count = self.count + 1
        image, heads = self.draw_boxes(image, scores, boxes, classes, im_width, im_height)
        return image, heads