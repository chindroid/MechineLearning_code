#coding=utf-8
import numpy as np
import math
from sklearn.datasets import load_boston#导入数据集的包
#from compiler.ast import flatten

#使用梯度下降法的线性回归
class LinearRegression2:
    def __init__(self,data,target):#data、target分别为点的坐标向量和对应的实际映射值
        self.data=data
        self.target=target
        self.theta=np.ndarray.flatten((np.zeros((1,len(data[0]))))).tolist() #参数都初始化为0
        for i in self.theta:
            i=i+1
        self.last_loss=self.loss_fun()+1#last_loss保存最近一次梯度下降之前的损失函数
    def hypothesis(self,X):#估计函数、其中所有的参数用theta表示
        result=0
        for i,j in zip(X,self.theta):
            result=result+i*j
        return result
    def loss_fun(self):#损失函数
        loss=0
        for i,j in zip(data,target):
            loss=loss+0.5*(self.hypothesis(i)-j)**2
        return loss
    def update_theta(self,step_length=0.01):#参数更新参数、step_length表示梯度下降的步长
        self.last_loss=self.loss_fun()
        new_theta=[]
        for j in range(len(self.theta)):#对每个参数theta都计算梯度
            gradient=0#表示梯度
            for i in range(len(self.data)):
                gradient=gradient+(target[i]-self.hypothesis(data[i]))*data[i][j]#计算梯度
            # print gradient
            new_t=self.theta[j]+step_length*gradient#批量梯度下降
            new_theta.append(new_t)
        self.theta=new_theta#更新参数向量theta
    def regress(self):
        step_count=0
        while True:
            step_count=step_count+1
            self.update_theta(0.00000001)
            if(step_count%100==0):
                print("step ",step_count)#显示步数
                print("Current theta:",self.theta,'\n',"loss:",self.loss_fun())#显示当前参数
            if math.fabs(self.loss_fun()-self.last_loss)<0.1:#前后loss变化小于阈值
                break
        print("After ",step_count,"steps,achieve iterative convergence!")
    def predict(self,X):
        result=0
        for i,j in zip(X,self.theta):
            result=result+i*j
        return result

if __name__ == '__main__':
    #从sklearn的数据集中获取相关向量数据集data和房价数据集target
    data,target=load_boston(return_X_y=True)
    LR=LinearRegression2(data,target)
    LR.regress()#线性回归模型的训练，梯度下降可能需要一段时间，如果先时间太长可以把参数保存到文件中，之后每次运行程序可以直接预测
    #选取一部分的样本用来验证最终回归模型的效果
    test_data=[]
    for i in range(len(data)):
        if i%17==0:
            test_data.append([data[i],target[i]])
    for i in test_data:
        print("real: ",i[1],"  estimate: ",LR.predict(i[0]))