import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt

timeDicts = np.load('./weight/pigCoughThermalv2Cnn/times.npy', allow_pickle='TRUE').item()
print(timeDicts)


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
plt.ylabel('Time(ms)')
modelList = ['ThermographicNet','Lenet5','AlexNet','DenseNet121','VGG16','VGG19','ResNet50','ResNet101','ResNet152']

index = 0
for i in timeDicts:
    j_count = 0
    x_v = -0.04 + index
    for j in timeDicts[i]:
        if j_count==5:

            plt.scatter(modelList[index],round(j*100,2),s=120,c=colorList[index],alpha=0.8)
        else:
            plt.scatter(x_v,round(j*100,2), c=colorList[index], s=150, alpha=0.2,marker='d')
            x_v = x_v + 0.03
        j_count += 1
    index += 1
plt.grid(axis='y',linestyle="--",alpha=0.4)
plt.savefig("v2_times.png", dpi=100, bbox_inches='tight')
plt.show()