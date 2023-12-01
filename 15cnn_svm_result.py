from IPython.display import display_markdown
import numpy as np
from prettytable import PrettyTable


table = PrettyTable()


table.field_names = ["模型及层", "Accuracy", "Precision", "Recall", "F1-score"]



all_model_name = ['Lenet-5','AlexNet','DenseNet121','自定义网络','Vgg16','Vgg19','ResNet50','ResNet101','ResNet152']
fc1_Accuracy = np.load('../weight/pigCoughThermalv2Cnn/lab2_fc1_Accuracy_result.npy', allow_pickle='TRUE').tolist()
fc1_Precision = np.load('../weight/pigCoughThermalv2Cnn/lab2_fc1_Precision_result.npy', allow_pickle='TRUE').tolist()
fc1_Recall = np.load('../weight/pigCoughThermalv2Cnn/lab2_fc1_Recall_result.npy', allow_pickle='TRUE').tolist()
fc1_F1 = np.load('../weight/pigCoughThermalv2Cnn/lab2_fc1_F1_result.npy', allow_pickle='TRUE').tolist()

fc2_Accuracy = np.load('../weight/pigCoughThermalv2Cnn/lab2_fc2_Accuracy_result.npy', allow_pickle='TRUE').tolist()
fc2_Precision = np.load('../weight/pigCoughThermalv2Cnn/lab2_fc2_Precision_result.npy', allow_pickle='TRUE').tolist()
fc2_Recall = np.load('../weight/pigCoughThermalv2Cnn/lab2_fc2_Recall_result.npy', allow_pickle='TRUE').tolist()
fc2_F1 = np.load('../weight/pigCoughThermalv2Cnn/lab2_fc2_F1_result.npy', allow_pickle='TRUE').tolist()

table.add_row([all_model_name[0]+" - FC1", fc1_Accuracy[0], fc1_Precision[0], fc1_Recall[0], fc1_F1[0]])
table.add_row([all_model_name[0]+" - FC2", fc2_Accuracy[0], fc2_Precision[0], fc2_Recall[0], fc2_F1[0]])

table.add_row([all_model_name[1]+" - FC1", fc1_Accuracy[1], fc1_Precision[1], fc1_Recall[1], fc1_F1[1]])
table.add_row([all_model_name[1]+" - FC2", fc2_Accuracy[1], fc2_Precision[1], fc2_Recall[1], fc2_F1[1]])

table.add_row([all_model_name[2]+" - FC1", fc1_Accuracy[2], fc1_Precision[2], fc1_Recall[2], fc1_F1[2]])

table.add_row([all_model_name[4]+" - FC1", fc1_Accuracy[4], fc1_Precision[4], fc1_Recall[4], fc1_F1[4]])
table.add_row([all_model_name[4]+" - FC2", fc2_Accuracy[3], fc2_Precision[3], fc2_Recall[3], fc2_F1[3]])

table.add_row([all_model_name[5]+" - FC1", fc1_Accuracy[5], fc1_Precision[5], fc1_Recall[5], fc1_F1[5]])
table.add_row([all_model_name[5]+" - FC2", fc2_Accuracy[4], fc2_Precision[4], fc2_Recall[4], fc2_F1[4]])

table.add_row([all_model_name[6]+" - FC1", fc1_Accuracy[6], fc1_Precision[6], fc1_Recall[6], fc1_F1[6]])

table.add_row([all_model_name[7]+" - FC1", fc1_Accuracy[7], fc1_Precision[7], fc1_Recall[7], fc1_F1[7]])

table.add_row([all_model_name[8]+" - FC1", fc1_Accuracy[8], fc1_Precision[8], fc1_Recall[8], fc1_F1[8]])

table.add_row([all_model_name[3]+" - FC1", fc1_Accuracy[3], fc1_Precision[3], fc1_Recall[3], fc1_F1[3]])
table.add_row([all_model_name[3]+" - FC2", fc2_Accuracy[2], fc2_Precision[2], fc2_Recall[2], fc2_F1[2]])
print(table)