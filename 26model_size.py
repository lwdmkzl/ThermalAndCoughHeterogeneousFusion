import os
import matplotlib.pyplot as plt

def get_FileSize(filePath):
    fsize = os.path.getsize(filePath)

model_baseDir = './weight/pigCoughThermalv2Cnn/'
all_model = ['Diy.cnn.cqt.best.h5','Lenet-5.cnn.cqt.best.h5','AlexNet.cnn.cqt.best.h5','DenseNet121.cnn.cqt.best.h5','Vgg16.cnn.cqt.best.h5','Vgg19.cnn.cqt.best.h5','ResNet50.cnn.cqt.best.h5','ResNet101.cnn.cqt.best.h5','ResNet152.cnn.cqt.best.h5']

svmmodel_baseDir = './weight/pigCoughThermalv2SVM/'
all_svm_model = ['lab3_fc1_acous.pkl','lab4_Lenet5_time.pkl','lab4_AlexNet_time.pkl','lab4_DenseNet121_time.pkl','lab4_Vgg16_time.pkl','lab4_Vgg19_time.pkl','lab4_ResNet50_time.pkl','lab4_ResNet101_time.pkl','lab4_ResNet152_time.pkl']
fSize = []
fSVMSize = []
for m in all_model:
    model_filepath = os.path.join(model_baseDir,m)
    fSize.append(get_FileSize(model_filepath))

for m in all_svm_model:
    model_filepath = os.path.join(svmmodel_baseDir,m)
    fSVMSize.append(get_FileSize(model_filepath))


font_path = "./TimesNewRoman.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.figure(figsize=(9,3.2))
ax = plt.gca()
ax.spines['right'].set_linewidth(1*0.5)
ax.spines['top'].set_linewidth(1*0.5)
ax.spines['left'].set_linewidth(1*0.5)
ax.spines['bottom'].set_linewidth(1*0.5)
plt.xlabel('Method')
plt.ylabel('Model Size(MB)')
modelList = ['ThermographicNet','Lenet5','AlexNet','DenseNet121','VGG16','VGG19','ResNet50','ResNet101','ResNet152']


index = 0
for i in modelList:
    print(fSize[index]+fSVMSize[index])
    plt.scatter(i,fSize[index]+fSVMSize[index],s=150,c=colorList[index],alpha=1)
    index += 1

plt.grid(axis='y',linestyle="--",alpha=0.4)
plt.savefig('v2_modelsize.png', dpi = 100, bbox_inches='tight')
plt.show()