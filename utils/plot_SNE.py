#t-Distributed Stochastic Neighbor Embedding (t-分布随机邻域嵌入, 简称 t-SNE) 是一种降维技术，特别适用于 高维数据集的可视化
# 注意，该方法目前仅支持12个类别及以下
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.datasets import load_breast_cancer
import time
import datetime
from utils.utils import mini_batch_deep_features
import torch

# print(y)
def plot_SNE(X_imgs,y,buffer):
    """
    参数详解：
    X: 单个任务的x_train，里面包含十个类的图片 5000*32*32*3，格式为numpy
    y: 单个任务的标签y
    """
    start = time.perf_counter()
    print("开始降维！")
    # 各种定义与赋值，以及将gpu上的数据转到numpy上
    model = buffer.model
    X_buffer_imgs = buffer.buffer_img
    y_buffer = buffer.buffer_label.cpu().detach().numpy()
    X_features= mini_batch_deep_features(model, torch.tensor(X_imgs), X_imgs.shape[0]).cpu().detach().numpy()

    X_buffer_features = mini_batch_deep_features(model,X_buffer_imgs,X_buffer_imgs.size(0))
    X_buffer_features = X_buffer_features.cpu().detach().numpy()
    # X_buffer_features————————y_buffer
    # X_features————————y

    X = np.concatenate((X_features, X_buffer_features), axis=0)

    # # 暂时遗弃的方法，有毒吧，靠这什么索引得搞到猴年马月去啊？
    # # 此处实现找到buffer中的样本在输入中的索引值
    # index = np.array([False]*X_features.shape[0])
    # for i in X_buffer_features:
    #     i_n = np.expand_dims(i, axis=0)
    #     i_n = np.repeat(i_n, X_features.shape[0], axis=0)
    #     print("i*n:",i_n)
    #     # print("result",X_features==np.array([list(i)]*X_buffer_features.shape[0]))
    #     # print("X_features:",X_features)
    #     result =  X_features==i_n
    #     print("sum_result",sum(result))
    #     result = np.all(i_n == True, axis=1)
        
    #     index = index + result
    # # print("index:",index)
    # # print("sum_true:",sum(index))
    # X = X_features
    
    tsne = do_tsne(X=X,save=True)

    # 构建绘图的颜色               
    cnames = {
    'aliceblue':            '#F0F8FF',
    'antiquewhite':         '#FAEBD7',
    'aqua':                 '#00FFFF',
    'aquamarine':           '#7FFFD4',
    'azure':                '#F0FFFF',
    'beige':                '#F5F5DC',
    'bisque':               '#FFE4C4',
    'black':                '#000000',
    'blanchedalmond':       '#FFEBCD',
    'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2',
    'brown':                '#A52A2A',
    'burlywood':            '#DEB887',
    'cadetblue':            '#5F9EA0',
    'chartreuse':           '#7FFF00',
    'chocolate':            '#D2691E',
    'coral':                '#FF7F50',
    'cornflowerblue':       '#6495ED',
    'cornsilk':             '#FFF8DC',
    'crimson':              '#DC143C',
    'cyan':                 '#00FFFF',
    'darkblue':             '#00008B',
    'darkcyan':             '#008B8B',
    'darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9',
    'darkgreen':            '#006400',
    'darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B',
    'darkolivegreen':       '#556B2F',
    'darkorange':           '#FF8C00',
    'darkorchid':           '#9932CC',
    'darkred':              '#8B0000',
    'darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F',
    'darkslateblue':        '#483D8B',
    'darkslategray':        '#2F4F4F',
    'darkturquoise':        '#00CED1',
    'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493',
    'deepskyblue':          '#00BFFF',
    'dimgray':              '#696969',
    'dodgerblue':           '#1E90FF',
    'firebrick':            '#B22222',
    'floralwhite':          '#FFFAF0',
    'forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF',
    'gainsboro':            '#DCDCDC',
    'ghostwhite':           '#F8F8FF',
    'gold':                 '#FFD700',
    'goldenrod':            '#DAA520',
    'gray':                 '#808080',
    'green':                '#008000',
    'greenyellow':          '#ADFF2F',
    'honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4',
    'indianred':            '#CD5C5C',
    'indigo':               '#4B0082',
    'ivory':                '#FFFFF0',
    'khaki':                '#F0E68C',
    'lavender':             '#E6E6FA',
    'lavenderblush':        '#FFF0F5',
    'lawngreen':            '#7CFC00',
    'lemonchiffon':         '#FFFACD',
    'lightblue':            '#ADD8E6',
    'lightcoral':           '#F08080',
    'lightcyan':            '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen':           '#90EE90',
    'lightgray':            '#D3D3D3',
    'lightpink':            '#FFB6C1',
    'lightsalmon':          '#FFA07A',
    'lightseagreen':        '#20B2AA',
    'lightskyblue':         '#87CEFA',
    'lightslategray':       '#778899',
    'lightsteelblue':       '#B0C4DE',
    'lightyellow':          '#FFFFE0',
    'lime':                 '#00FF00',
    'limegreen':            '#32CD32',
    'linen':                '#FAF0E6',
    'magenta':              '#FF00FF',
    'maroon':               '#800000',
    'mediumaquamarine':     '#66CDAA',
    'mediumblue':           '#0000CD',
    'mediumorchid':         '#BA55D3',
    'mediumpurple':         '#9370DB',
    'mediumseagreen':       '#3CB371',
    'mediumslateblue':      '#7B68EE',
    'mediumspringgreen':    '#00FA9A',
    'mediumturquoise':      '#48D1CC',
    'mediumvioletred':      '#C71585',
    'midnightblue':         '#191970',
    'mintcream':            '#F5FFFA',
    'mistyrose':            '#FFE4E1',
    'moccasin':             '#FFE4B5',
    'navajowhite':          '#FFDEAD',
    'navy':                 '#000080',
    'oldlace':              '#FDF5E6',
    'olive':                '#808000',
    'olivedrab':            '#6B8E23',
    'orange':               '#FFA500',
    'orangered':            '#FF4500',
    'orchid':               '#DA70D6',
    'palegoldenrod':        '#EEE8AA',
    'palegreen':            '#98FB98',
    'paleturquoise':        '#AFEEEE',
    'palevioletred':        '#DB7093',
    'papayawhip':           '#FFEFD5',
    'peachpuff':            '#FFDAB9',
    'peru':                 '#CD853F',
    'pink':                 '#FFC0CB',
    'plum':                 '#DDA0DD',
    'powderblue':           '#B0E0E6',
    'purple':               '#800080',
    'red':                  '#FF0000',
    'rosybrown':            '#BC8F8F',
    'royalblue':            '#4169E1',
    'saddlebrown':          '#8B4513',
    'salmon':               '#FA8072',
    'sandybrown':           '#FAA460',
    'seagreen':             '#2E8B57',
    'seashell':             '#FFF5EE',
    'sienna':               '#A0522D',
    'silver':               '#C0C0C0',
    'skyblue':              '#87CEEB',
    'slateblue':            '#6A5ACD',
    'slategray':            '#708090',
    'snow':                 '#FFFAFA',
    'springgreen':          '#00FF7F',
    'steelblue':            '#4682B4',
    'tan':                  '#D2B48C',
    'teal':                 '#008080',
    'thistle':              '#D8BFD8',
    'tomato':               '#FF6347',
    'turquoise':            '#40E0D0',
    'violet':               '#EE82EE',
    'wheat':                '#F5DEB3',
    'white':                '#FFFFFF',
    'whitesmoke':           '#F5F5F5',
    'yellow':               '#FFFF00',
    'yellowgreen':          '#9ACD32'}
    cnames = list(cnames)


    print("已完成降维！所用时间：",time.perf_counter()-start)
    print("==================================")
    # # tsne 归一化， 这一步可做可不做
    # x_min, x_max = tsne.min(0), tsne.max(0)
    # tsne_norm = (tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(20, 20))

    # 此处将所有输入的数据进行了降维
    for i in set(y):
        tsne_0 = tsne[:X_features.shape[0],:][y == i]
        print("tsne_0",tsne_0)    
        print("tsne_0.shape",tsne_0.shape)    
        plt.scatter(tsne_0[:, 0], tsne_0[:, 1], 3, color=cnames[i], label=i,marker= "o")
        # marker= "o","v","^","<","s","P"
    

    for i in set(y_buffer):
        tsne_1 = tsne[X_features.shape[0]:,:][y_buffer == i]
        print("tsne_1",tsne_1)    
        print("tsne_1.shape",tsne_1.shape)
        plt.scatter(tsne_1[:, 0], tsne_1[:, 1], 12, color=cnames[i], label=i,marker= "v")

    # tsne_normal[i, 0]为横坐标，X_norm[i, 1]为纵坐标，1为散点图的面积， color给每个类别设定颜色
    plt.legend(loc='upper left')
    plt.show()
    curr_time = datetime.datetime.now()
    name = "SNE_"+str(curr_time)[:20]+".png"
    plt.savefig('/home/lzj/online-continual-learning-main/pictures/'+name)

    # TSNE 方法使用
def do_tsne(X,save=True):
    # n_components:嵌入空间的维度
    # perpexity:混乱度，优化过程中考虑临近点的多少，默认为30，建议在5-50之间
    # learning rate 学习率，表示梯度下降的快慢，默认为200，建议取值在10-1000之间
    # n_iter 迭代次数，默认为1000，自定义设置时应保证大于250
    # early_exaggeration 表示嵌入空间簇间距大小，默认为12
    TSNE = manifold.TSNE(n_components=2, init='pca',early_exaggeration=25,n_iter = 5000 ,random_state=2)
    tsne = TSNE.fit_transform(X)
    if save:
        np.savetxt("tsne.csv", tsne, delimiter=",", newline=",")
    return tsne


if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    y[y==0]=15
    y[y==1]=16
    plot_SNE(X,y)