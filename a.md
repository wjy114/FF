#问题求解大作业
  姚迪熙 518021910367
##数据与环境
   运行环境 Ubuntu1804LTS
   设备环境 Dell G7 (CPU Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz,GPU NVIDIA Corporation GP106M [GeForce GTX 1060 Mobile] (rev a1))
   编译环境 g++,gcc version 7.4.0 (Ubuntu 7.4.0-1ubuntu1~18.04.1) 	      
   standard C++14
   dataset    San Francisco Crime Classification [Source](https://www.kaggle.com/kaggle/san-francisco-crime-classification)
   本次作业已整合成脚本，检查时可在命令行下运行
 
       $bash ./run

就可以测试所有的功能

建议将数据集放在`./datas`文件夹下，或在任务运行是输入训练集的绝对路径(具体操作在任务2中说明)
任务3和任务4运行需要先运行任务2的程序
训练好的模型在`./FINAL_SOLUTION/model.txt`
如果运行./run不行请运行`bash ./run`
如果出现`./run:...:权限不够`请尝试将对应任务源代码所在文件夹下的.o和可执行文件删除后再尝试运行
---
##Task 1 数据输入
本部分源代码地址在 ` ./depends` 文件夹下的 `csvreader.h`;`csvreader.cpp`
###任务描述
择取一个数据集并为之设计合适的数据结构，以便高效地存取相关数据
###解决方法及结果
这一部分的灵感来源于python 下 pandas 包，在C++中我手动实现了一下。
我用了两种简单的数据结构来存储我的数据，第一种结构是使得数据更加的直观
	
	csvfile
而在这个类中最核心的部分是data这个变量
	
	map<string,vector<string>> data
使用了一个map类型，对于每个pair它的索引就是列名比如date,weekday,PDdistrict ,etc.，而第二个则是一个向量，这个向量保存了每一列所有的数据。

而第二个数据结构则是便于之后进行data science的处理。但是这个数据结构没有普适性，只能用于我的数据集。
在`./depends` 文件夹下的`utils.h`;`utils.cpp`；
我定义一个这样一个类型

	loader
最核心的部分是

	vector<vector<double>> data;
可以把它视作是一个二维数组或者矩阵，除去最后一列每一列表示一个feature，最后一列表示label也就是我要处理的目标，是否是恶性犯罪，而每一行则表示一个identity，也就是每个嫌疑人。
比如

	vector<vector<double>> data(rows,vector<double>(cols));

就是总共有rows组数据，feature+label总共有cols个。
行列的含义也可以反过来，但是在这个project中之后处理是按这个规则进行的。
在我接下来的处理过程中我都是使用这样一个二维的vector
有了这样一个矩阵来管理数据，接下来处理就方便了许多。
---
##Task2  脏数据的发现与处理
测试可运行

	$bash ./run
	$1
在运行后会询问数据集所在的路径，既可以选择输入绝对路径也可以将数据集放在`./data`文件夹下，并输入default都可以完成文件的加载，，如果没有找到文件会提示such file does not exist

这一部分的源代码在` ./FE/FE.cpp`，而主要完成任务的地方则是`./depends`文件夹的 `filter.h`;`filter.cpp`;`utils.h`;`utils.cpp`
###任务描述
你的数据集中是否存在脏数据？请定义一些标准发现并处理这些脏数据
###解决方法
####脏数据定义
我定义了三类脏数据

1. useless feature
2. feature相同，label不同
3. false district
####脏数据处理
对于三类脏数据我的处理方法分别是
#####useless feature
原始数据中有很多的feature，但是很多是没有什么用处的，总共剔除了三个abundant feature  1.descrip是对案件的描述，每一个事件description都不一样，所以没有什么参考意义。2.resolution 对于案件的解决办法，在预测案件的时候是不可能有解决办法因而此特征对于任务无任何作用 3.address过于详细，如果作为训练模型的feature反而会使模型难以收敛，训练效果差，并且对于预测案件无参考意义。将没有用的feature去除即可。
最后总结下来有用的feature有

* year
* month
* day
* hour 
* minute 
* week
* PDdistrict
* 经度
* 纬度

这里有一个问题需要考虑，是不是week和日期的信息重复，经纬度和地区的信息重复，可能乍一看确实不需要应该作为冗余特征，但是其实不是，因为星期的信息确实可以由日期来导出，但是这个导出的函数也是一个非线性(mod 函数)的函数。因而在模型训练过程中如果删除星期这个特征会增加模型非线性的负担，甚至在不好的参数下会使得问题不是一个凸问题，所以我决定不给自己挖坑所以还是加进去了。当然我这样做会有一个问题，最后会过拟合，不过最后我发现我并没有遇到这个问题就没有回过来看。

这一部分主要是由`utils.h`中的`loader::from_csv_data`函数完成

#####feature相同 ，label不同
要是数据集里面feature相同，label不同的数据很多，模型最后很容易不收敛，所以这是个很严重的问题

基于基本的原则，建设安全和平社区，对于相同的feature按照严重的情况处理，否则很容易让危险份子逃脱。

这一部分由`tools.h`中的`wash`函数完成

#####false distric
发现数据集里面有一些数据的经纬度是错误的，纬度大于38甚至有90那肯定不是在旧金山，以及经度的错误比如说-122度以西等，已经不是旧金山了，把它们去除即可。

这一部分由`tools.h`中的`false_district`函数完成

---

处理好feature以后对于label按照是否是严重的犯罪，进行分类 1表示严重犯罪 0 表示一般犯罪(国情不同，这一标准按照我自己的理解来，如果实际应用应该由犯罪学专家来处理，不过和本课程内容无关，所以只能当一个示范)

| 英文名 | 中文名 | 类别|
|-|-|-|
| ﻿ARSON | 纵火 | 1|
| ASSAULT |暴力攻击|1|
| BAD CHECKS |空支票|0|
| BRIBERY | 贿赂|1|
| BURGLARY | 入室盗窃|1|
| DISORDERLY CONDUCT | 带头骚乱|1|
| DRIVING UNDER THE INFLUENCE | 毒品酗酒后驾车|1|
| DRUG/NARCOTIC | 毒品|1|
| DRUNKENNESS | 酒精犯罪|1|
| EMBEZZLEMENT | 贪污|0|
| EXTORTION | 勒索|1|
| FAMILY OFFENSES | 家暴|0|
| FORGERY/COUNTERFEITING | 伪造|0|
| FRAUD | 诈骗|0|
| GAMBLING | 赌博|0|
| KIDNAPPING | 绑架儿童|1|
| LARCENY/THEFT | 偷窃|1|
| LIQUOR LAWS | 酒精违法|0|
| LOITERING | 街头滞留罪|0|
| MISSING PERSON | 失踪人口|1|
| NON-CRIMINAL | 非犯罪|0|
| OTHER OFFENSES | 其他攻击|1|
| PORNOGRAPHY/OBSCENE MAT | 色情|1|
|PROSTITUTION | 卖淫|1|
| RECOVERED VEHICLE | 车丢失但找回 | 0|
| ROBBERY | 抢劫|1|
| RUNAWAY | 逃跑|1|
| SECONDARY CODES | 次级犯罪|0|
| SEX OFFENSES FORCIBLE | 暴力性性攻击|1|
| SEX OFFENSES NON FORCIBLE | 非暴力性性攻击|1|
| STOLEN PROPERTY | 财产盗窃|0|
| SUICIDE | 自杀|0|
| SUSPICIOUS OCC | 可疑虚拟货币|0|
| TREA | 入侵或者闲逛在工业财产附近|0|
| TRESPASS | 入侵|1|
| VANDALISM | 故意毁坏文物|1|
| VEHICLE THEFT | 车辆盗窃|1|
| WARRANTS | 保释|0|
| WEAPON LAWS | 武器攻击|1|

---
###结果
最后我们分别按照之前定义好的数据结构保存成适合C++，和python处理的 .csv文件，结果在`./datas`文件夹下的`dataset.txt`和`dataset_py.csv`

---
##Task3  统计与可视化
###任务描述
对数据进行统计(至少从三个维度统计数据)，并使用FLTK可视化统计后的数据
###环境
fltk 1.3.5
###解决方法及结果
有三个统计数据的维度
1. 对比了13年来旧金山市在一个礼拜中不同星期严重犯罪和一般犯罪的犯罪数量
对于之前设定好的数据结构只要判断 data[5],与data[9]并计算总数就可以完成统计。
代码在 `./visualize/main1.cpp` 中
测试可运行

	$bash ./run
	$21
	
对于犯罪总数统计， 统计结果如下

Crimes | severe | non-severe 
-|-|-
Monday | 63898 | 49200
Tuesday | 66422 | 50061
Wednesday | 68949 | 51388
Thursday | 66420 | 50298
Friday | 70681 | 54405
Saturday | 66436 | 52110
Sunday | 61341 | 47234



然后使用FLTK进行可视化结果如下
![图一](./visualize/1.png)

2.统计了13年来每年严重犯罪和一般犯罪的数量，并做成动态的折线图
代码在`./visualize/main2.cpp`中
测试可运行
	
	$bash  ./run
	$ 22
统计结果如下

Year |  severe | non-severe
-|-|-
2003 | 41922 | 69999
2004 | 42624 | 69400
2005 | 40755 | 67479
2006 | 37743 | 66116
2007 | 36796 | 63858
2008 | 39957 | 66054
2009 | 39019 | 64439
2010 | 35441 | 61336
2011 | 33682 | 61340
2012 | 35091 | 66085
2013 | 33916 | 68647
2014 | 34806 | 68692
2015 | 12395 | 25398



然后使用FLTK进行可视化结果如下
(做成了动画）
![图2](./visualize/2.gif)

3.统计了13年来发生的所有案件发生的位置
代码在`./visualize/main3.cpp` 中
测试可运行

	$bash ./run
	$23
统计用核心代码如下

	//地址结构
	struct location{
 		location(double X,double Y):x(X),y(Y){}
 		double x,y;
	};
	//统计经纬度，sl严重案件，nsl一般案件
 	for(auto x:dataset)
		if(x[9])sl[(int)(x[0]-2003)].push_back(location((x[7]-122000)/1000,(x[8]+37700)/1000));
		else nsl[(int)(x[0]-2003)].push_back(location((x[7]-122000)/1000,(x[8]+37700)/1000));

然后使用FLTK进行可视化结果如下
![图三](./visualize/3.png)

---
##Task4 趋势预测或分类
###任务描述
设定目标，并使用课程中介绍的线性回归技术或者自学新的技术实现目标。你如何说明所用方法的准确性？
###解决方法
####目标
根据病人体检的身体特征预测其是否患有心血管疾病
####训练集与测试集
前90%的数据作为训练集，后10%数据作为测试集
####评价标准
评价标准的实现在`./depends/tools.h`文件中的`ac`函数
对于使用的方法，采用了四个标准，分别是

* recall 召回率
* percision 精确率
* accuracy 准确率
* latency 运行时间

接下来先解释下它们是什么
真实指该数据在数据集中label
预测是模型根据feature计算出来的答案

|   | 真实 1 | 真实 0 |
|-|-|-|
| 预测1 | TP | FP | 
| 预测0 |FN |TN |
 
 $$recall = \frac{TP}{FN+TP}$$
 $$percision = \frac{TP}{FP+TP}$$
 $$accuracy = \frac{TP+TN}{FN+TP+TN+FP}$$

接下来解释一下为什么要引入这几个标准

1. 准确率的引入是显而易见的，用来判断我的模型所预测的内容是不是正确的，准确率越高那么模型预测的必然是越精准的
2. 但是 ,对于一个全部均匀分布的样本如果我全部猜1或者全部猜0，那么accuracy也可以达到50%左右，因而accuracy就失去了意义
3. 因而引入recall和percision以确保模型的有效准确
4. 在这个数据集中recall表示确实严重犯罪并且预测也是严重犯罪，因而需要尽量高，percision表示预测是严重犯罪并且实际是严重犯罪，准确率表示正确的预测。

因为train和test都不是用的固定的训练集和数据集而是随机进行0.9和0.1的划分，所以运行结果和报告可能会略有一些差别
####实现方法
#####方法1 线性回归
代码在`./LR/LR.cpp`
测试可运行

	$bash ./run
	$31

使用简单线性回归
先对数据使用归一化，这一部分的代码在`tools.h`中，使用了atan和l1范数归一化方法使得feature之间相差不大
方法梯度下降，学习率0.001 ,当损失变化<1e-7时停止运算

	产生随机参数  vecotr<double> theta(feature_num+1);
	计算损失，损失函数使用MSE
	按照泰勒公式 
	先计算假设函数和真实值差hx
	对于每个feature计算偏导数
	double res=0;
    	for(int i=0;i<row;++i)
    	{
        	res+=hx[i]*train[i][x];
    	}
    	res/=(datanum*2);
    更新参数
    temp[i]=theta[i]-learning_rate*f(i);
    重新计算损失

计算出所有参数以后就是结果
最终结果

	recall 1
	percision 0.566099
	accuracy 0.566099
	using time  457.879s



#####方法2 xgboost
代码在`./xgbst/xgbst.c`
测试可运行

	$bash ./run
	$32

既然是个二分类问题我就想到了XGBoost利用回归数和决策数的一种集成学习方法，这一个方法没有手动实现，主要是通过调库 xgboost

	$ git clone --recursive https://github.com/dmlc/xgboost  
	$ cd xgboost 
	$ make -j4  
	$ ./xgboost 
生成C++的API
经过一些简单的调参

	safe_xgboost(XGBoosterSetParam(booster, "booster", "gbtree"));
  	safe_xgboost(XGBoosterSetParam(booster, "objective", "multi:softmax"));
  	safe_xgboost(XGBoosterSetParam(booster, "num_class", "2")); 
  	safe_xgboost(XGBoosterSetParam(booster, "min_child_weight", "6"));
  	safe_xgboost(XGBoosterSetParam(booster, "gamma", "0"));
  	safe_xgboost(XGBoosterSetParam(booster, "max_depth", "4"));
  	safe_xgboost(XGBoosterSetParam(booster, "subsample", "1"));
  	safe_xgboost(XGBoosterSetParam(booster, "colsample_bytree", "1"));
  	safe_xgboost(XGBoosterSetParam(booster, "learning_rate", "0.05"));
  	safe_xgboost(XGBoosterSetParam(booster, "n_estimators", "100"));
  	safe_xgboost(XGBoosterSetParam(booster, "scale_pos_weight", "1"));
  	safe_xgboost(XGBoosterSetParam(booster, "reg_alpha", "0"));
  	safe_xgboost(XGBoosterSetParam(booster, "reg_lambda", "1"));
  	safe_xgboost(XGBoosterSetParam(booster, "seed", "45"));
  	safe_xgboost(XGBoosterSetParam(booster, "eta","0.01"));
  
 由于库本身的限制原因,只能开辟数组过大会栈溢出，所以选取了200000个数据点z作为展示，由于最终此方法没有被纳入到最终方法中，所以就不考虑是否使用全部数据的问题了。
最后的结果如下
	
 	recall 0.668737
	percision 0.564918
	accuracy 0.52315
	using time  13.7394s



#####方法3 neural network
代码在`./NN/NN.cpp`以及`./NN/BPnet.h`;`./NN/BPnet.cpp`
测试可运行

	$bash ./run
	$33

手动实现了前馈神经网络，使用梯度下降的方法
学习率 0.01 当损失函数变化<1e-5时结束训练(视作收敛)
准备工作
先对数据使用归一化，这一部分的代码在`tools.h`中，使用了atan和l1范数归一化方法使得feature之间相差不大
设计了两种激活函数分别是Relu和sigmoid
按照课上学习到的知识，全链接参每一个节点都和下一层结点有所链接
设计了三种数据结构
输入结点：包含计算值，对于下一个结点的权值，和正反向传播用到的偏差
中间结点：包含计算值，权值，偏置量，以及它们和正确答案的偏差
输出结点：包含计算值，真值，偏置量，以及其偏差

算法思路

	先全部随机初始化
	while(dif>1e-5){
	计算损失 MSE
	正向传播，按照路径把所有计算值全部计算出来
	反向传播，按照逆向路径计算所有计算值和真值之间的偏差 delta 
	参数更新，按照正向路径，按照delta，更新参数 weight/bias-=lr*weight/bias
	}

经过一定过程的调参，最终确定 使用2层隐藏层，每层210个结点
并设计了相应的函数保存模型，加载模型便于训练好的模型多次使用
最终结果如下
	
	recall 1
	percision 0.566501
	accuracy 0.566501
	using time  289.145s

###结果
####最终方法
代码在`./FINAL_SOLUTION/final_solution.cpp`
测试可运行

	$bash ./run
	$3

最终结合之前的方法，我发现我忽略了一个问题，一般有些很厉害的侦探在调查案件的时候有的时候并不是依靠线索，而是有的时候会按照部分的经验判断，因此在这个project中，如果有案件和之前发生过的案件一样的化，就可以直接分类。所以在最后首先对于数据查找是否有以往的案例，如果有一样的就直接判断
所以最后有两步

* `map<vector<double>,double> search_map;`搜索表搜索是否有相同案件
* 神经网络预测

最终运行五次并取平均值的结果如下

	recall 0.601305
	percision 0.810787
	accuracy 0.643064
	using time  6.27321s
	
最终可以看到召回率还是比较客观的，其他两个指标也还是比较不错的，时间也用的比较短，最终是实现了我的目标

##补充
对于各种模型的训练，我同样也写了python的代码，有兴趣可以在./run中尝试运行
有任何问题欢迎联系 Jimmyyao18@sjtu.edu.cn
最后衷心感谢老师和助教的付出！

##License
© 姚迪熙 ，2019
