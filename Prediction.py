import numpy as np
from lightgbm import LGBMRegressor
import numpy
import logging
import asyncio
import random

# 测试数据
test_data=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
11999,9390,13481,14795,15845,15271,14686,11054,10395]
# 初始化
logger = logging.getLogger("Predictor")
# 预测类
class predictor:
    def __init__(self, look_back=15, n_models=20, n_estimators=100, learning_rate = 0.05):
        '''
        :param look_back: 将之前的多少个时间点作为特征
        :param n_estimators: 多少个树
        :param learning_rate: 学习率
        '''
        self.look_back=look_back
        self.n_models = n_models
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.models = []
        # 数据集
        self.raw_data = None
        # 训练数据
        self.data_X=None
        self.data_y=None
    # 数据集构建
    async def prepare_dataset(self,data,split_proportion=0.2):
        loop = asyncio.get_event_loop()
        self.raw_data = numpy.array(data).astype(numpy.float64)
        # 空列表，准备存放我们需要的数据集
        X, y=[],[]
        # 如果数据量不够
        if len(data)<self.look_back:
            raise Exception("Error: No enough data for training!")
        def sync_dataset_prepare():
            for i in range(self.look_back,len(data)):
                # 特征
                features = data[i-self.look_back:i]
                # 标签，即当前值
                label = data[i]
                X.append(features)
                y.append(label)
        await loop.run_in_executor(None,sync_dataset_prepare)
        # 整数型
        int_data_X = numpy.array(X)
        int_data_y = numpy.array(y)
        # 强转为float64
        self.data_X = int_data_X.astype(np.float64)
        self.data_y = int_data_y.astype(np.float64)
    # 训练函数
    async def train(self):
        '''
        :param data: 数据
        :param future_prediction_total: 总共要预测的分钟数
        '''
        loop = asyncio.get_event_loop()
        # 同步训练函数
        def train_sync():
            for i in range(self.n_models):
                # 随机模型
                model = LGBMRegressor(
                    n_estimators=self.n_estimators+i*10,
                    learning_rate=self.lr+0.001*i,
                    random_state = 42+i,
                    verbose = -1
                )
                indices = numpy.random.choice(len(self.data_X), len(self.data_X), replace=True)
                model.fit(self.data_X[indices],self.data_y[indices])
                self.models.append(model)
        await loop.run_in_executor(None,train_sync)
        logger.info("model trained successfully!")
    # 预测函数
    async def predict(self, data, total_future_count):
        loop = asyncio.get_event_loop()
        # 同步预测函数
        def sync_prediction():
            current_window = self.raw_data[-self.look_back:].copy()
            # 添加噪声，防止自回归模型退化
            current_window += numpy.random.normal(0, 0.02, current_window.shape)
            predictions = []
            for _ in range(total_future_count):
                # 预测下一个点
                next_prediction = random.choice(self.models).predict(current_window.reshape(1, -1))[0]
                predictions.append(next_prediction)
                # 上一个点出队，预测点入队
                current_window = numpy.append(current_window[1:], next_prediction)
            return predictions
        # 准备数据集
        await self.prepare_dataset(data)
        # 训练
        await self.train()
        # 预测
        predictions = await loop.run_in_executor(None, sync_prediction)
        return predictions
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    pred = predictor()
    print(loop.run_until_complete(pred.predict(test_data,120)))
