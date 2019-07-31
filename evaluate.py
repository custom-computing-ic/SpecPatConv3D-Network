from train import net
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from helper import showClassTable, maybeExtract

parser = ArgumentParser()
parser.add_argument('--patch_size', type=int, default=5)
parser.add_argument('--data', type=str, default='Indian_pines', help='Indian_pines or Salinas or KSC')

number_of_band = {'Indian_pines': 2, 'Salinas': 2, 'KSC': 2, 'Botswana': 1}

def evaluate(opt):

    _, _, TEST = maybeExtract(opt.data, opt.patch_size)
    test_data, test_label = TEST[0], TEST[1]
    HEIGHT = test_data.shape[1]
    WIDTH = test_data.shape[2]
    CHANNELS = test_data.shape[3]
    N_PARALLEL_BAND = number_of_band[opt.data]
    NUM_CLASS = test_label.shape[1]

    graph = tf.Graph()
    with graph.as_default():

        # Define Model entry placeholder
        img_entry = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANNELS])
        img_label = tf.placeholder(tf.uint8, shape=[None, NUM_CLASS])

        # Get true class from one-hot encoded format
        image_true_class = tf.argmax(img_label, axis=1)

        # Dropout probability for the model
        prob = tf.placeholder(tf.float32)

        # Network model definition
        model = net(img_entry, prob, HEIGHT, WIDTH, CHANNELS, N_PARALLEL_BAND, NUM_CLASS)

        # Cost Function
        final_layer = model['dense3']

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_layer,
                                                                       labels=img_label)
            cost = tf.reduce_mean(cross_entropy)

        # Optimisation function
        with tf.name_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)

        # Model Performance Measure
        with tf.name_scope('accuracy'):
            predict_class = model['predict_class_number']
            correction = tf.equal(predict_class, image_true_class)

        accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))
        saver = tf.train.Saver()

        with tf.Session(graph=graph) as session:
            saver.restore(session, tf.train.latest_checkpoint('./Trained_model/'))
            def test(t_data, t_label, test_iterations=1, evalate=False):
                assert test_data.shape[0] == test_label.shape[0]
                y_predict_class = model['predict_class_number']
                # OverallAccuracy, averageAccuracy and accuracyPerClass
                overAllAcc, avgAcc, averageAccClass = [], [], []
                for _ in range(test_iterations):

                    pred_class = []
                    for t in tqdm(t_data):
                        t = np.expand_dims(t, axis=0)
                        feed_dict_test = {img_entry: t, prob: 1.0}
                        prediction = session.run(y_predict_class, feed_dict=feed_dict_test)
                        pred_class.append(prediction)

                    true_class = np.argmax(t_label, axis=1)
                    conMatrix = confusion_matrix(true_class, pred_class)

                    # Calculate recall score across each class
                    classArray = []
                    for c in range(len(conMatrix)):
                        recallScore = conMatrix[c][c] / sum(conMatrix[c])
                        classArray += [recallScore]
                    averageAccClass.append(classArray)
                    avgAcc.append(sum(classArray) / len(classArray))
                    overAllAcc.append(accuracy_score(true_class, pred_class))

                averageAccClass = np.transpose(averageAccClass)
                meanPerClass = np.mean(averageAccClass, axis=1)

                showClassTable(meanPerClass, title='Class accuracy')
                print('Average Accuracy: ' + str(np.mean(avgAcc)))
                print('Overall Accuracy: ' + str(np.mean(overAllAcc)))

            test(test_data, test_label, test_iterations=1)

if __name__ == '__main__':
    option = parser.parse_args()
    evaluate(option)