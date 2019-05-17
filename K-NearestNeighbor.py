#coding=utf-8
import math
#定义鸢尾花的数据类
class Iris:
    data=[]
    label=[]
    pass
#定义一个读取莺尾花数据集的函数
def load_dataset(filename="Iris_train.txt"):
    f=open(filename)
    line=f.readline().strip()
    propty=line.split(',')#属性名
    dataset=[]#保存每一个样本的数据信息
    label=[]#保存样本的标签
    while line:
        line=f.readline().strip()
        if(not line):
            break
        temp=line.split(',')
        content=[]
        for i in temp[0:-1]:
            content.append(float(i))
        dataset.append(content)
        label.append(temp[-1])
    total=Iris()
    total.data=dataset
    total.label=label
    return total#返回数据集
 
#定义一个Knn分类器类
class KnnClassifier:
    def __init__(self,k,type="Euler"):#初始化的时候定义正整数K和距离计算方式
        self.k=k
        self.type=type
        self.dataloaded=False
    def load_traindata(self,traindata):#加载数据集
        self.data=traindata.data
        self.label=traindata.label
        self.label_set=set(traindata.label)
        self.dataloaded=True#是否加载数据集的标记
 
    def Euler_dist(self,x,y):# 欧拉距离计算方法，x、y都是向量
        sum=0
        for i,j in zip(x,y):
            sum+=math.sqrt((i-j)**2)
        return sum
    def Manhattan_dist(self,x,y):#曼哈顿距离计算方法，x、y都是向量
        sum=0
        for i,j in zip(x,y):
            sum+=abs(i-j)
        return sum
    def predict(self,temp):#预测函数，读入一个预测样本的数据，temp是一个向量
        if(not self.dataloaded):#判断是否有训练数据
            print "No train_data load in"
            return
        distance_and_label=[]
        if(self.type=="Euler"):#判断距离计算方式，欧拉距离或者曼哈顿距离
            for i,j in zip(self.data,self.label):
                dist=self.Euler_dist(temp,i)
                distance_and_label.append([dist,j])
        else:
            if(self.type=="Manhattan"):
                for i,j in zip(self.data,self.label):
                    dist=self.Manhattan_dist(temp,i)
                    distance_and_label.append([dist,j])
            else:
                print "type choice error"
        #获取K个最邻近的样本的距离和类别标签
        neighborhood=sorted(distance_and_label,cmp=lambda x,y : cmp(x[0],y[0]))[0:self.k]
        neighborhood_class=[]
        for i in neighborhood:
            neighborhood_class.append(i[1])
        class_set=set(neighborhood_class)
        neighborhood_class_count=[]
        print "In k nearest neighborhoods:"
        #统计该K个最邻近点中各个类别的个数
        for i in class_set:
            a=neighborhood_class.count(i)
            neighborhood_class_count.append([i,a])
            print "class: ",i,"   count: ",a
        result=sorted(neighborhood_class_count,cmp=lambda x,y : cmp(x[1],y[1]))[-1][0]
        print "result: ",result
        return result#返回预测的类别
 
if __name__ == '__main__':
    traindata=load_dataset()#training data
    testdata=load_dataset("Iris_test.txt")#testing data
    #新建一个Knn分类器的K为20，默认为欧拉距离计算方式
    kc=KnnClassifier(20)
    kc.load_traindata(traindata)
    predict_result=[]
    #预测测试集testdata中所有待预测样本的结果
    for i,j in zip(testdata.data,testdata.label):
        predict_result.append([i,kc.predict(i),j])
    correct_count=0
    #将预测结果和正确结果进行比对，计算该次预测的准确率
    for i in predict_result:
        if(i[1]==i[2]):
            correct_count+=1
    ratio=float(correct_count)/len(predict_result)
    print "correct predicting ratio",ratio
