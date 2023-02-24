import os
import cv2
import numpy as np
from queue import Queue

class LineStateClassifier:
    """docstring for ."""

    def __init__(self, logger, params):
        """
        """
        self._logger = logger
        
        self._class_names = params['class_names']

        self._threshold = params['threshold']
        self._pred_q = Queue(maxsize = params['queue_size'])
        if params['on_edge']:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=os.path.expanduser(params['model_path']))
        else:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=os.path.expanduser(params['model_path']))
        self.interpreter.allocate_tensors()


    def preprocess(self, mask, input_shape):
        """
        """
        mask = cv2.bitwise_not(mask)
        # cv2.imshow("thr", mask)
        # cv2.waitKey(10)
        out_img = cv2.resize(mask, (input_shape[1], input_shape[2]))
        out_img = np.array(np.reshape(out_img, input_shape), dtype=np.float32)
        out_img = out_img / 255.
        return out_img

    def predict(self, mask, debug=False):
         # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        # self._logger.info(f"model input details: {input_details}")
        input_data = self.preprocess(mask, input_shape)
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        pred_sum = np.zeros(6)
        if self._pred_q.full():
            self._pred_q.get()
        self._pred_q.put(output_data[0])
        for elem in list(self._pred_q.queue):
            pred_sum += elem

        average_query_pred = pred_sum/self._pred_q.qsize()

        if any(average_query_pred > self._threshold):
            pred_index = np.argmax(average_query_pred)
            pred_class = self._class_names[pred_index]
        else:
            pred_class = 'Low accuracy'
        if debug:
            self._logger.info(f'current queue size {self._pred_q.qsize()}')
            self._logger.debug(f'logits: {output_data[0]}')
            self._logger.info(f'The predicted class is: {pred_class}')
        return pred_class, output_data[0]

if __name__ == '__main__':
    params = dict()
    params['input_shape'] = (64,64,1)
    params['number_classes'] = 6
    params['threshold'] = 0.65
    params['queue_size'] = 5
    params['class_names'] = ['straight', 'x', 'T', 'left', 'right', 'end']
    params['model_path'] = '/home/mohamed/robolympix/gray_line_classifier.tflite'
    params['on_edge'] = False
    params['debug'] = True
    line_classifier = LineStateClassifier(params)
    img = cv2.imread('../media/T2.jpg')

    test_img = line_classifier.preprocess(img, params['input_shape'])
    pred_class , _ = line_classifier.predict(test_img)
