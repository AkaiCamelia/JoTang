#!/usr/bin/env python
# coding: utf-8

# In[7]:


#导入所需的package
import pickle
import gzip

import random
import numpy as np

#导入MNIST数据集
def load_data():
   #打开并读取位于特定位置的mnist数据集  
   f = gzip.open('mnist.pkl.gz', 'rb')   
   training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
   f.close()
   #返回所需要的数据供训练   
   return (training_data, validation_data, test_data)

#将数据转化为适合训练的形式
def load_data_wrapper():
   #导入MNST数据集
   tr_d, va_d, te_d = load_data()
   #将训练集转化为数组
   training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
   training_results = [vectorized_result(y) for y in tr_d[1]]
   training_data = zip(training_inputs, training_results)
   #对验证集进行相同操作
   validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
   validation_data = zip(validation_inputs, va_d[1])
   #对测试集进行相同操作
   test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
   test_data = zip(test_inputs, te_d[1])
   
   return (training_data, validation_data, test_data)
#将输出值转化为一个10维向量
def vectorized_result(j):
  
   e = np.zeros((10, 1))
   e[j] = 1.0
   return e



#主体
class Network(object):
   def __init__(self, sizes):
       #定义层，偏置和权重
       self.num_layers = len(sizes)
       self.sizes = sizes
       #随机初始化偏置和权重
       self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
       self.weights = [np.random.randn(y, x)
                       for x, y in zip(sizes[:-1], sizes[1:])]
   #前馈计算
   def feedforward(self, a):
      
       for b, w in zip(self.biases, self.weights):
           a = sigmoid(np.dot(w, a)+b)
       return a
#实现随机梯度下降（SGD）算法用于训练模型
   def SGD(self, training_data, epochs, mini_batch_size, eta,
           test_data=None):
       #将数据从数据集中取出
       training_data = list(training_data)
       n = len(training_data)
       
       if test_data:
           test_data = list(test_data)
           n_test = len(test_data)
       
       for j in range(epochs):
           #随机化处理训练数据
           random.shuffle(training_data)
           #划分minibatch
           mini_batches = [
               training_data[k:k+mini_batch_size]
               for k in range(0, n, mini_batch_size)]
           for mini_batch in mini_batches:
               self.update_mini_batch(mini_batch, eta)
           #打印结果，以供参考
           if test_data:
               print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
           else:
               print("Epoch {} complete".format(j))
#对minibatch进行反向传播计算以便更新权重和偏差
   def update_mini_batch(self, mini_batch, eta):
       #用来存储权重和置
       nabla_b = [np.zeros(b.shape) for b in self.biases]
       nabla_w = [np.zeros(w.shape) for w in self.weights]
       #计算梯度
       for x, y in mini_batch:
           delta_nabla_b, delta_nabla_w = self.backprop(x, y)
           #存储
           nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
           nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
       #更新权重和偏置
       self.weights = [w-(eta/len(mini_batch))*nw
                       for w, nw in zip(self.weights, nabla_w)]
       self.biases = [b-(eta/len(mini_batch))*nb
                      for b, nb in zip(self.biases, nabla_b)]


#反向传播过程
   def backprop(self, x, y):
       
       nabla_b = [np.zeros(b.shape) for b in self.biases]
       nabla_w = [np.zeros(w.shape) for w in self.weights]
       
       activation = x
       activations = [x]
       zs = []
       for b, w in zip(self.biases, self.weights):
           z = np.dot(w, activation)+b
           zs.append(z)
           #激活函数
           activation = sigmoid(z)
           activations.append(activation)
       
       delta = self.cost_derivative(activations[-1], y) * \
           sigmoid_prime(zs[-1])
       nabla_b[-1] = delta
       nabla_w[-1] = np.dot(delta, activations[-2].transpose())
     
       for l in range(2, self.num_layers):
           z = zs[-l]
           sp = sigmoid_prime(z)
           delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
           nabla_b[-l] = delta
           nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
       return (nabla_b, nabla_w)

   
   #对计算的过程进行评估，返回正确的个数
   def evaluate(self, test_data):
      
       test_results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in test_data]
       return sum(int(x == y) for (x, y) in test_results)

   
   #输出激活函数的偏导数向量
   def cost_derivative(self, output_activations, y):
       
       return (output_activations-y)

#定义sigmoid函数
def sigmoid(z):
  
   return 1.0/(1.0+np.exp(-z))

#定义sigmoid导数
def sigmoid_prime(z):

   return sigmoid(z)*(1-sigmoid(z))

#加载数据并创建网络
training_data, validation_data, test_data = load_data_wrapper()
# 输入层784个神经元，隐含层30个神经元，输出层10个神经元
net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 2.0, test_data=test_data)


# In[ ]:





# In[ ]:




