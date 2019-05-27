from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from argparse import ArgumentParser
from model import *
import tensorflow as tf
import numpy as np
from helper import showClassTable, maybeExtract, maybeDownloadOrExtract, GroundTruthVisualise
from helper import get_available_gpus
import os

from tqdm import tqdm

number_of_band = {'Indian_pines': 2, 'Salinas': 2, 'KSC': 2, 'Botswana': 1}

# get_available_gpus()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# GPU_DEVICE_IDX = '1'
model_directory = os.path.join(os.getcwd(), 'Trained_model/')

parser = ArgumentParser()
parser.add_argument('--data', type=str, default='Indian_pines', help='Indian_pines or Salinas or KSC')
parser.add_argument('--epoch', type=int, default=650, help='Epochs')
parser.add_argument('--batch_size', type=int, default=50, help='Mini batch at training')
parser.add_argument('--patch_size', type=int, default=5)
parser.add_argument('--device', type=str, default='CPU')


# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

def net(statlieImg, prob, HEIGHT, WIDTH, CHANNELS, N_PARALLEL_BAND, NUM_CLASS):
    # A dictionary expect to contain graph nodes
    # Arguement: Image place holder, Dropout probability,

    sequence = {}
    sequence['input'] = tf.reshape(statlieImg, [-1, HEIGHT, WIDTH, CHANNELS])

    # Block 1 Conv1 layer
    with tf.variable_scope('conv1'):
        layer = sequence['input']
        layer = create_conv_2dlayer(input=layer,
                                    filter_size=1,
                                    num_output_channel=CHANNELS,
                                    relu=True,
                                    padding='valid')
        sequence['conv1'] = layer
    # Tensor shape = N * 5 * 5 * 220

    with tf.variable_scope('parallelProcess'):
        layer = sequence['conv1']

        with tf.variable_scope('reshape3d'):
            layer = tf.reshape(layer, [-1, HEIGHT, WIDTH, CHANNELS, 1])

        with tf.variable_scope('split'):
            layer = tf.split(layer, num_or_size_splits=N_PARALLEL_BAND, axis=3)

        with tf.variable_scope('segmentation', reuse=tf.AUTO_REUSE):  # Enable parameter sharing

            for index, l in enumerate(layer):
                with tf.variable_scope('layer1'):
                    layer1 = create_conv_3dlayer(input=l,
                                                 filter_width=2,
                                                 filter_height=2,
                                                 filter_depth=9,
                                                 stride=2,
                                                 num_output_channels=1,
                                                 relu=True)

                with tf.variable_scope('layer2'):
                    layer2 = create_conv_3dlayer(input=layer1,
                                                 filter_width=3,
                                                 filter_height=3,
                                                 filter_depth=5,
                                                 stride=1,
                                                 num_output_channels=3,
                                                 relu=True)

                with tf.variable_scope('layer3'):
                    layer3 = create_conv_3dlayer(input=layer2,
                                                 filter_width=2,
                                                 filter_height=2,
                                                 filter_depth=3,
                                                 stride=2,
                                                 num_output_channels=6,
                                                 relu=True)

                with tf.variable_scope('layer4'):
                    layer4 = create_conv_3dlayer(input=layer3,
                                                 filter_width=1,
                                                 filter_height=1,
                                                 filter_depth=3,
                                                 stride=1,
                                                 num_output_channels=10,
                                                 relu=True)

                layer5, _ = flatten_layer(layer4)

                if not index:
                    stack = tf.concat([layer5], axis=1)
                else:
                    stack = tf.concat([stack, layer5], axis=1)

        sequence['parallel_end'] = stack

    with tf.variable_scope('dense1'):
        layer = sequence['parallel_end']
        layer, number_features = flatten_layer(layer)
        layer = fully_connected_layer(input=layer,
                                      num_inputs=number_features,
                                      num_outputs=120,
                                      activation='relu')
        layer = tf.nn.dropout(x=layer, keep_prob=prob)
        sequence['dense1'] = layer

    with tf.variable_scope('dense3'):
        layer = sequence['dense1']
        layer = fully_connected_layer(input=layer,
                                      num_inputs=120,
                                      num_outputs=NUM_CLASS)
        sequence['dense3'] = layer

    y_predict = tf.nn.softmax(sequence['dense3'])
    sequence['class_prediction'] = y_predict
    sequence['predict_class_number'] = tf.argmax(y_predict, axis=1)

    return sequence


def main(opt):

    # Load MATLAB data that contains data and labels
    TRAIN, VALIDATION, TEST = maybeExtract(opt.data, opt.patch_size)

    # Extract data and label from MATLAB file
    training_data, training_label = TRAIN[0], TRAIN[1]
    validation_data, validation_label = VALIDATION[0], VALIDATION[1]
    test_data, test_label = TEST[0], TEST[1]

    print('\nData shapes')
    print('training_data shape' + str(training_data.shape))
    print('training_label shape' + str(training_label.shape) + '\n')
    print('validation_data shape' + str(validation_data.shape))
    print('validation_label shape' + str(validation_label.shape) + '\n')
    print('test_data shape' + str(test_data.shape))
    print('test_label shape' + str(test_label.shape) + '\n')

    SIZE = training_data.shape[0]
    HEIGHT = training_data.shape[1]
    WIDTH = training_data.shape[2]
    CHANNELS = training_data.shape[3]
    N_PARALLEL_BAND = number_of_band[opt.data]
    NUM_CLASS = training_label.shape[1]

    EPOCHS = opt.epoch
    BATCH = opt.batch_size

    graph = tf.Graph()
    with graph.as_default():
        # Define Model entry placeholder
        img_entry = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANNELS])
        img_label = tf.placeholder(tf.uint8, shape=[None, NUM_CLASS])

        # Get true class from one-hot encoded format
        image_true_class = tf.argmax(img_label, axis=1)

        # Dropout probability for the model
        prob = tf.placeholder(tf.float32)

        # BASS-net model definition
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

        # Checkpoint Saver
        saver = tf.train.Saver()
        with tf.Session(graph=graph) as session:

            # Uncomment the below line to view the TensorBoard
            '''
            writer = tf.summary.FileWriter("Network-logs/", session.graph)
            '''
            '''
            if os.path.isdir(model_directory):
                saver.restore(session, 'Trained_model/')
            '''
            session.run(tf.global_variables_initializer())

            def test(t_data, t_label, test_iterations=1):

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

            def train(num_iterations, train_batch_size=50):

                maxValidRate = 0
                for i in range(num_iterations + 1):

                    print('Optimization Iteration: ' + str(i))

                    for x in range(int(SIZE / train_batch_size) + 1):
                        train_batch = training_data[x * train_batch_size: (x + 1) * train_batch_size]
                        train_batch_label = training_label[x * train_batch_size: (x + 1) * train_batch_size]
                        feed_dict_train = {img_entry: train_batch, img_label: train_batch_label, prob: 0.5}
                        _, loss_val = session.run([optimizer, cross_entropy], feed_dict=feed_dict_train)

                    if i % 15 == 0:
                        acc = session.run(accuracy, feed_dict={img_entry: validation_data,
                                                               img_label: validation_label,
                                                               prob: 1.0})
                        print('Model Performance, Validation accuracy: ', acc * 100)
                        if maxValidRate < acc:
                            location = i
                            maxValidRate = acc
                        print('Maximum validation accuracy: ', acc, ' at epoch ', location)
                        test(validation_data, validation_label, 1)

            def count_param():
                total_parameters = 0
                for variable in tf.trainable_variables():
                    shape = variable.get_shape()
                    variable_parameters = 1
                    for dim in shape:
                        variable_parameters *= dim.value
                    total_parameters += variable_parameters
                print('Trainable parameters: ' + '\033[92m' + str(total_parameters) + '\033[0m')

            count_param()
            # Train model
            train(num_iterations=EPOCHS, train_batch_size=BATCH)
            saver.save(session, model_directory)

            # Test model
            test(test_data, test_label, test_iterations=1)
            print('End session ' + str(opt.data))

if __name__ == '__main__':
    option = parser.parse_args()
    main(option)
