#coding=utf-8
import  math
import copy

class Iris:
    attribute=[]
    data=[]
    label=[]
    pass

def load_dataset(filename="Iris_train.txt"):
    f=open(filename)
    line=f.readline().strip()
    propty=line.split(',')[:-1]#属性名

    total=Iris()
    total.attribute=propty
    dataset=[]#保存每一个样本的数据信息+
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
    total.data=dataset
    total.label=label
    return total

class DCTree:
    attribute=[]
    data=[]
    label=[]
    partition_value=[]
    data_loaded=False
    trained=False
    attribute2num={}
    num2attribute={}
    total_D=[]

    tree=[]#用来存放节点的树结构

    def __init__(self):
        pass
    def load_traindata(self,traindata,label,attribute):#traindata为一个二维列表
        self.data=traindata
        self.label=label
        self.attribute=attribute
        self.data_loaded=True
        r=range(len(attribute))
        self.attribute2num=dict(zip(attribute,r))
        for i,j in zip(traindata,label):
            self.total_D.append([i,j])

    #计算信息熵
    def Ent(self,D):#D为带标签的数组
        label=[]
        for i in D:
            label.append(i[1])
        label_set=set(label)
        sum_=0
        for i in label_set:
            temp=label.count(i)/float(len(label))
            if temp==1:
                continue
            else:
                sum_-=temp*math.log(2,temp)#迭代计算信息熵
        return sum_

    #基于二分的方式，选出当前数据集D中最所有属性的最优划分点，返回一个字典保存attribute_中属性和对应划分点的值的键值对
    def compute_partition_value(self,D,attribute_):
        attribute_2_attribute_value={}
        for i in attribute_:
            temp=[]
            for j in D:
                temp.append(j[0][self.attribute2num[i]])
            temp=sorted(temp)#将每种属性的值放在一个列表中并排序
            partition_temp=[]
            for j in range(len(temp)-1):
                partition_temp.append((temp[j]+temp[j+1])/2)
            max_parti_value=partition_temp[0]
            max_gain=-999999

            D_a=[]
            for j in D:
                D_a.append([j[0][self.attribute2num[i]],j[1]])
            ent_d=self.Ent(D_a)#计算D整体样本集的信息熵
            for j in partition_temp:
                D1=[]
                D2=[]
                for k in D_a:
                    if(k[0]<j):
                        D1.append(k)
                    else:
                        D2.append(k)
                temp_gain=ent_d-(len(D1)*self.Ent(D1)/float(len(D_a))+len(D2)*self.Ent(D2)/float(len(D_a)))
                if temp_gain>max_gain:
                    max_parti_value=j
            attribute_2_attribute_value[i]=max_parti_value
        return attribute_2_attribute_value

    #计算信息增益
    def Gain(self,D,attribute_,a):#D为带标签的数组,[[属性值,标签],...],a为划分属性属性
        attribute_2_parti_value_dict=self.compute_partition_value(D,attribute_)
        max_gain=-999999
        a_id=self.attribute2num[a]
        D_a=[]
        for i in D:
            D_a.append([i[0][a_id],i[1]])
        ent_d=self.Ent(D_a)#计算D整体样本集的信息熵
        D1=[]
        D2=[]
        for j in D_a:
            if(j[0]<attribute_2_parti_value_dict[a]):
                 D1.append(j)
            else:
                 D2.append(j)
        gain_value=ent_d-(len(D1)*self.Ent(D1)/float(len(D_a))+len(D2)*self.Ent(D2)/float(len(D_a)))
        return gain_value
    #为当前数据集构造决策树，递归过程
    def generate_dctree(self,D,attributes_):#D为带标签的样本集，attributes为还未划分的属性
        tree=[]#定义子决策树
        # print D
        label=[]
        for i in D:
            label.append(i[1])
        print label
        set_label=set(label)
        #递归返回条件：1.当前数据集中所有对象属于同一个类别；2.没有剩余的划分属性了；3.当前数据集为空；满足一个便可将当前数据集作为一个叶子节点
        if len(set_label)==1:#当前数据集中所有对象属于同一个类别，该类别就作为当前叶子节点的类别
            print "leaf_1: ",set_label[0]
            return ["leaf node",set_label[0]]
        print "attributes_:",attributes_
        if len(attributes_)==0:#若没有在可以用来划分的属性，取该叶子中所有数据集中类别最多的一类
            max_count=0
            leaf_label=""
            for i in set_label:
                print "i:",i
                if label.count(i)>max_count:
                    print label.count(i)
                    max_count=label.count(i)
                    leaf_label=i
                print "leaf_2: ",leaf_label
            return ["leaf node",leaf_label]
        if len(D)==0:#若当前数据集为空则返回空
            print "NULL"
            return
        max_a=-9999
        best_attribute=""
        attribute_2_parti_value_dict=self.compute_partition_value(D,attributes_)
        for a in attributes_:
            temp=self.Gain(D,attributes_,a)
            if temp>max_a:
                max_a=temp
            best_attribute=a
        #定义用于递归的子数据集

        sub_D1=[]
        sub_D2=[]
        parti_value=attribute_2_parti_value_dict[best_attribute]
        print "best_attribute:",best_attribute,"pativalue:",parti_value
        a_id=self.attribute2num[best_attribute]
        #将当前数据集根据当前最优属性的划分点分成两个子数据集
        for i in D:
            if i[0][a_id]<parti_value:
                sub_D1.append(i)
            else:
                sub_D2.append(i)
        print "sub_D1:",sub_D1
        print "sub_D2:",sub_D2
        node_info="not leaf node"
        tree.append(node_info)#是否是叶子节点信息
        tree.append(best_attribute)#若不是叶子节点，当前节点的划分属性
        tree.append(parti_value)#当前节点的划分属性的划分值
        new_attribute_=copy.copy(attributes_)
        new_attribute_.remove(best_attribute)#移除当前节点使用的属性，剩余的属性作为子决策树的属性集
        sub_tree1=self.generate_dctree(sub_D1,new_attribute_)
        sub_tree2=self.generate_dctree(sub_D1,new_attribute_)
        tree.append(sub_tree1)
        tree.append(sub_tree2)
        return tree #返回当前决策树
    #训练函数
    def train(self):
        if not self.data_loaded:
            print "No data loaded!"
            return
        #生成决策树
        self.tree=self.generate_dctree(self.total_D,self.attribute)
        self.trained=True
    #预测函数
    def predict(self,testdata):#testdata为一个一维列表
        if not self.trained:
            print "Please train DCT first!"
            return
        curr_tree=copy.deepcopy(self.tree)
        #决策树从根节点开始的沿着路径向下寻找对应叶节点
        while curr_tree[0]!="leaf node":
            curr_attribute=curr_tree[1]
            a_id=self.attribute2num[curr_attribute]
            if testdata[a_id]<curr_tree[2]:
                curr_tree=curr_tree[3]#左子树
            else:
                curr_tree=curr_tree[4]#右子树
        result=curr_tree[1]
        return result
if __name__ == '__main__':

    dataset=load_dataset("Iris_train.txt")
    print dataset
    dct=DCTree()#读入样本属性集
    dct.load_traindata(dataset.data,dataset.label,dataset.attribute)
    dct.train()
    print dct.tree
    #加载测试集
    testset=load_dataset("Iris_test.txt")
    predict_result=[]
    for i in testset.data:
        predict_result.append(dct.predict(i))
    print predict_result
    print testset.label