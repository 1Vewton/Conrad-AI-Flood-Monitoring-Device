import lightgbm
import numpy
import logging
import asyncio
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 测试数据
# 具有上升趋势和波动的测试数据
test_data = [10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
11999,9390,13481,14795,15845,15271,14686,11054,10395]


def generate_trendy_data():
    # 基础上升趋势
    x = numpy.arange(300)
    trend = 0.1 * x
    # 添加噪声
    noise = numpy.random.normal(0, 1, 300)
    data = trend + noise
    return data
# 初始化
logger = logging.getLogger("Predictor")
# 处理数据集
def create_features(window, full_series):
    # 特征
    features = []
    # 滞后特征
    features.extend([window[0],
                     window[int(len(window)/2)],
                     window[len(window)-1]])
    # 统计特征
    features.extend([
        # 平均
        numpy.mean(window),
        # 标准差
        numpy.std(window),
        # 变化量
        window[-1] - window[0]
    ])
    # 差分特征
    if len(window) > 1:
        diff_features = numpy.diff(window)
        # 差分特征插入
        features.extend([
            # 平均变化率
            numpy.mean(diff_features),
            # 变化稳定性
            numpy.std(diff_features)
        ])
    # 趋势特征
    if len(window)>=3:
        x = numpy.arange(len(window))
        # 斜率
        slope = numpy.polyfit(x, window, 1)[0]
        features.append(slope)
    # 全局趋势特征
    if len(window)<=len(full_series):
        long_mean = numpy.mean(full_series)
        short_mean = numpy.mean(window)
        features.extend([
            # 绝对差异
            short_mean - long_mean,
            # 相对比例
            short_mean/(long_mean+1e-8)
        ])
        # 全局线性趋势
        global_x = numpy.arange(len(full_series))
        global_slope = numpy.polyfit(global_x, full_series, 1)[0]
        features.append(global_slope)
    # 动量特征
    if len(window) >= 4:
        momentum = window[-1]-2*window[-2]+window[-3]
        features.append(momentum)
    return features
# 预测类
class predictor:
    def __init__(self, look_back=15, n_models=20, n_estimators=100, learning_rate = 0.05):
        '''
        :param look_back: 将之前的多少个时间点作为特征
        :param n_estimators: 多少个树
        :param learning_rate: 学习率
        '''
        self.look_back=look_back
        # 模型数量
        self.n_models = n_models
        # 基础参数
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.models = [None]*self.n_models
        # 模型分数
        self.models_score = [None]*self.n_models
        # 模型权重
        self.models_weight = []
        # 数据集
        self.raw_data = None
        # 训练数据
        self.data_X=None
        self.data_y=None
        # 测试数据
        self.test_X = None
        self.test_y = None
    # 数据集构建
    async def prepare_dataset(self,data, split_proportion=0.8):
        loop = asyncio.get_event_loop()
        self.raw_data = data
        # 空列表，准备存放我们需要的数据集
        X, y=[],[]
        # 如果数据量不够
        if len(data)<self.look_back:
            raise Exception("Error: No enough data for training!")
        def sync_dataset_prepare():
            for i in range(self.look_back,len(data)):
                # 特征
                features = create_features(data[i-self.look_back:i], data)
                # 标签，即当前值
                label = data[i]
                X.append(features)
                y.append(label)
            logger.info("Finished dataset prepare. ")
            # 整数型
            int_data_X = numpy.array(X)
            int_data_y = numpy.array(y)
            # 分割数据集
            test_data_prop = int(len(int_data_y)*split_proportion)
            print(test_data_prop)
            train_data_y = int_data_y[:test_data_prop]
            test_data_y = int_data_y[test_data_prop+1:]
            train_data_X = int_data_X[:test_data_prop]
            test_data_X = int_data_X[test_data_prop+1:]
            # 强转为float64
            self.data_X = train_data_X.astype(numpy.float64)
            self.data_y = train_data_y.astype(numpy.float64)
            self.test_X = test_data_X.astype(numpy.float64)
            self.test_y = test_data_y.astype(numpy.float64)
        await loop.run_in_executor(None,sync_dataset_prepare)
    # 获取模型权重
    def get_model_weights(self):
        # 使用MSE的倒数作为权重基础
        mse_scores = [score['mse'] for score in self.models_score]
        min_mse = numpy.min(mse_scores)
        if min_mse == 0.0:
            min_mse = 1e-8
        # softmax-like
        quality_scores = [min_mse/(score['mse']+1e-8) for score in self.models_score]
        exp_scores = numpy.exp(quality_scores-numpy.max(quality_scores))
        self.models_weight = exp_scores/numpy.sum(exp_scores)
    # 训练函数
    async def train(self):
        '''
        :param data: 数据
        :param future_prediction_total: 总共要预测的分钟数
        '''
        loop = asyncio.get_event_loop()
        # 同步训练函数
        def train_sync(i:int):
            # 测试集
            test_dataset = lightgbm.Dataset(data=self.test_X, label=self.test_y)
            # 训练集
            indices = numpy.random.choice(len(self.data_X), len(self.data_X), replace=True)
            train_dataset = lightgbm.Dataset(data=self.data_X[indices], label=self.data_y[indices])
            # 模型参数
            params = {
                "objective": "regression",
                "n_estimators": self.n_estimators + i * 10,
                "metric": "l2",
                "num_leaves": 31 + (i % 5) * 5,
                "learning_rate": self.lr + 0.001 * i,
                "random_state": 42 + i,
                "verbose": -1
            }
            # 各种模型，避免出现退化现象
            model = lightgbm.train(params=params, train_set=train_dataset, valid_sets=[test_dataset])
            self.models[i] = model
            logger.info(f"Trained model: {i + 1}/{self.n_models}")
            # 评估模型
            test_result = model.predict(self.test_X)
            mse = mean_squared_error(self.test_y, test_result)
            mae = mean_absolute_error(self.test_y, test_result)
            # 模型分数
            score = {
                'mse': mse,
                'mae': mae,
                'rmse': numpy.sqrt(mse),
                'model_idx': i
            }
            self.models_score[i] = score
        # 异步处理
        async def train_async(i:int):
            await loop.run_in_executor(None,train_sync,i)
        tasks = []
        for i in range(self.n_models):
            tasks.append(asyncio.create_task(train_async(i)))
        await asyncio.wait(tasks)
        logger.info("model trained successfully!")
        self.get_model_weights()
    # 单个模型预测
    def ensemble_predict(self, total_future_count, midx:int):
        # 结果的prediction
        predictions = []
        current_window = self.raw_data[-self.look_back:].copy()
        feature_window = numpy.array(create_features(current_window, self.raw_data))
        for i in range(total_future_count):
            # 预测下一个点
            next_prediction = self.models[midx].predict(feature_window.reshape(1, -1))[0]
            predictions.append(next_prediction)
            # 上一个点出队，预测点入队
            current_window = numpy.append(current_window[1:], next_prediction)
            feature_window = numpy.array(create_features(current_window, self.raw_data))
        return numpy.array(predictions)*self.models_weight[midx]
    # 预测函数
    async def predict(self, data, total_future_count:int):
        loop = asyncio.get_event_loop()
        # 预测
        self.predictions = numpy.zeros(total_future_count)
        # 同步预测函数
        def predict_sync(i:int):
            return self.ensemble_predict(total_future_count, i)
        # 异步预测函数
        async def predict_async(i:int):
            res = await loop.run_in_executor(None, predict_sync, i)
            self.predictions = self.predictions+res
        # 准备数据集
        await self.prepare_dataset(data)
        # 训练
        await self.train()
        # 异步预测
        tasks = []
        for i in range(self.n_models):
            tasks.append(asyncio.create_task(predict_async(i)))
        await asyncio.wait(tasks)
        return self.predictions
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    pred = predictor()
    test_data = generate_trendy_data()
    res = loop.run_until_complete(pred.predict(test_data, 120))
    loop.run_until_complete(pred.train())
    print(res)
    print(len(test_data))
    plt.plot(numpy.append(numpy.array(test_data),res))
    plt.show()
    print(test_data)