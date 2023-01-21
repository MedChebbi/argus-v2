import cv2
import numpy as np
from queue import Queue

class LineStateClassifier:
    """docstring for ."""

    def __init__(self, logger, params):
        """
        """
        self._logger = logger
        
        self.class_names = params['class_names']
        self.num_classes = len(params['class_names'])

        self.shape = params['input_shape']
        self.threshold = params['threshold']
        self.pred_q = Queue(maxsize = params['queue_size'])
        self.debug = params['debug']
        if params['on_edge']:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=params['model_path'])
        else:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=params['model_path'])
        self.interpreter.allocate_tensors()

    @staticmethod
    def preprocess(gray_img, input_shape):
        img = cv2.resize(gray_img, (input_shape[0],input_shape[1]))
        img = np.array(np.reshape(img,input_shape),
                                  dtype=np.uint8)
        if input_shape[2]==3:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        out_img = (np.expand_dims(img, axis = 0))
        out_img = out_img/255.
        return out_img

    def predict(self, image):
         # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        #set random np array to test
        #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        preprocessed_image = self.preprocess(image, self.shape)
        input_data = np.array(preprocessed_image, dtype=np.float32)
        self.interpreter.set_tensor(input_details[0]['index'], input_data)

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        pred_sum = np.zeros(6)
        if self.pred_q.full():
            self.pred_q.get()
        self.pred_q.put(output_data[0])
        for elem in list(self.pred_q.queue):
            pred_sum += elem

        average_query_pred = pred_sum/self.pred_q.qsize()

        if any(average_query_pred > self.threshold):
            pred_index = np.argmax(average_query_pred)
            pred_class = self.class_names[pred_index]
        else:
            pred_class = 'Low accuracy'
        if self.debug:
            self._logger.info("current queue size" ,self.pred_q.qsize())
            self._logger.info('logits: ' ,output_data[0])
            self._logger.info('The predicted class is:', pred_class)
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
