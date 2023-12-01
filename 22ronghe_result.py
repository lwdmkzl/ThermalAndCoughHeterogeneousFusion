import numpy as np
from prettytable import PrettyTable

ronghe_lab1 = np.load('../weight/pigCoughThermalv2Cnn/lab_ronghe_lab1.npy', allow_pickle='TRUE').tolist()
ronghe_lab2 = np.load('../weight/pigCoughThermalv2Cnn/lab_ronghe_lab2.npy', allow_pickle='TRUE').tolist()
ronghe_lab3 = np.load('../weight/pigCoughThermalv2Cnn/lab_ronghe_lab3.npy', allow_pickle='TRUE').tolist()
ronghe_lab4 = np.load('../weight/pigCoughThermalv2Cnn/lab_ronghe_lab4.npy', allow_pickle='TRUE').tolist()
ronghe_lab5 = np.load('../weight/pigCoughThermalv2Cnn/lab_ronghe_lab5.npy', allow_pickle='TRUE').tolist()

table = PrettyTable()

table.field_names = ["特征", "Accuracy", "Precision", "Recall", "F1-score"]

table.add_row(["声源特征", ronghe_lab1[0], ronghe_lab1[1], ronghe_lab1[2], ronghe_lab1[3]])
table.add_row(["深度特征", ronghe_lab2[0], ronghe_lab2[1], ronghe_lab2[2], ronghe_lab2[3]])
table.add_row(["深度特征层融合", ronghe_lab3[0], ronghe_lab3[1], ronghe_lab3[2], ronghe_lab3[3]])
table.add_row(["声源特征 + 深度特征FC1+FC2融合", ronghe_lab4[0], ronghe_lab4[1], ronghe_lab4[2], ronghe_lab4[3]])
table.add_row(["声源特征 + 深度特征最佳层（F1）", ronghe_lab5[0], ronghe_lab5[1], ronghe_lab5[2], ronghe_lab5[3]])
print(table)