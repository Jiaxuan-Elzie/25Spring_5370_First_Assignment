import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

class FVIWithData:
    """
    带数据输入的 FVI
    """
    def __init__(self, price_data, cash_rate=0.03, T=10, gamma=1.0):
        """
        Parameters:
        -----------
        price_data : pd.DataFrame
            历史价格数据，index为日期，columns为资产代码
        cash_rate : float
            现金利率
        T : int
            时域长度
        gamma : float
            折扣因子
        """
        self.price_data = price_data
        self.cash_rate = cash_rate
        self.T = T
        self.gamma = gamma
        
        # 从历史数据估计收益分布
        returns = price_data.pct_change().dropna()
        self.mu = returns.mean().values
        self.cov = returns.cov().values
        self.n_assets = len(self.mu)
        
        print(f"数据加载完成: {len(price_data)} 个交易日, {self.n_assets} 个资产")
        print(f"收益均值: {self.mu}")
        
    def sample_returns(self, n_samples=100):
        """
        从历史估计的分布中采样收益
        """
        return np.random.multivariate_normal(self.mu, self.cov, size=n_samples)
    
    def terminal_utility(self, W, alpha=1.0):
        """
        CARA 终端效用
        """
        return -np.exp(-alpha * W)
    
    def train(self, n_states=500, n_actions=200, n_samples=100):
        """
        训练 FVI
        """
        # 采样状态
        states = self._sample_states(n_states)
        features = self._feature_matrix(states)
        
        value_functions = [None] * (self.T + 1)
        
        for t in range(self.T - 1, -1, -1):
            print(f"拟合时间步 {t}/{self.T-1}")
            targets = []
            
            for W, p in states:
                best_value = -np.inf
                
                # 搜索最优动作
                actions = self._sample_actions(n_actions)
                for action in actions:
                    # 计算期望值
                    expected = self._compute_expected(W, p, action, 
                                                       value_functions[t+1], 
                                                       n_samples)
                    if expected > best_value:
                        best_value = expected
                
                targets.append(best_value)
            
            # 拟合值函数
            model = LinearRegression()
            model.fit(features, targets)
            value_functions[t] = model
        
        self.value_functions = value_functions
        print("训练完成")
        
    def _sample_states(self, n):
        """采样状态空间"""
        states = []
        for _ in range(n):
            W = np.random.uniform(0.5, 3.0)
            p = np.random.dirichlet([1] * (self.n_assets + 1))
            p_risk = p[:self.n_assets]
            states.append((W, p_risk))
        return states
    
    def _feature_matrix(self, states):
        """构造特征矩阵"""
        features = []
        for W, p in states:
            # 特征: 1, W, 各资产仓位, 现金仓位
            feat = [1.0, W]
            feat.extend(p)
            feat.append(1.0 - np.sum(p))
            features.append(feat)
        return np.array(features)
    
    def _sample_actions(self, n):
        """采样动作空间"""
        actions = []
        for _ in range(n):
            p = np.random.dirichlet([1] * (self.n_assets + 1))
            actions.append(p[:self.n_assets])
        return np.array(actions)
    
    def _compute_expected(self, W, p, action, V_next, n_samples):
        """计算期望值"""
        p_new = action / np.sum(action)  # 归一化
        
        # 采样收益
        returns = self.sample_returns(n_samples)
        
        # 计算新财富
        portfolio_returns = returns @ p_new  # (n_samples,)
        W_new = W * (1 + portfolio_returns)
        
        # 计算值
        if V_next is not None:
            values = []
            for w in W_new:
                feat = [1.0, w]
                feat.extend(p_new)
                feat.append(1.0 - np.sum(p_new))
                values.append(V_next.predict([feat])[0])
            values = np.array(values)
        else:
            values = self.terminal_utility(W_new)
        
        return values.mean()
    
    def get_action(self, W, p, t):
        """获取最优动作"""
        if t >= self.T:
            return None
        
        # 搜索最优动作
        best_action = None
        best_value = -np.inf
        
        for action in self._sample_actions(200):
            expected = self._compute_expected(W, p, action, 
                                              self.value_functions[t+1], 
                                              n_samples=50)
            if expected > best_value:
                best_value = expected
                best_action = action
        
        return best_action

    def backtest_cumulative(self, W0=1.0, p0=None):
        """
        在整个历史序列上跑一遍最优策略，记录每步的财富和累计收益
        """
        if not hasattr(self, 'value_functions'):
            raise RuntimeError("请先调用 train() 进行训练")

        if p0 is None:
            p0 = np.ones(self.n_assets) / (self.n_assets + 1)

        hist_returns = self.price_data.pct_change().dropna().values
        n_days = len(hist_returns)
        dates = self.price_data.index[1:]  # 去掉第一天

        W = W0
        p = p0.copy()
        wealth_list = [W0]
        reward_list = [0.0]
        cum_reward = 0.0

        for day in range(n_days):
            t = day % self.T  # 周期性重置时间步
            action = self.get_action(W, p, t)
            if action is None:
                wealth_list.append(W)
                reward_list.append(cum_reward)
                continue

            p_new = action / np.sum(action)
            cash_weight = 1.0 - np.sum(p_new)

            r = hist_returns[day]
            portfolio_return = np.dot(p_new, r) + cash_weight * (self.cash_rate / 252)
            step_reward = portfolio_return  # 单步奖励 = 组合收益率
            cum_reward += step_reward

            W = W * (1 + portfolio_return)
            p = p_new
            wealth_list.append(W)
            reward_list.append(cum_reward)

        wealth_arr = np.array(wealth_list)
        cum_return_pct = (wealth_arr / W0 - 1) * 100
        cum_reward_arr = np.array(reward_list) * 100  # 百分比

        # 绘图
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        ax1 = axes[0]
        ax1.plot(dates, cum_return_pct[1:], 'b-', linewidth=1.5)
        ax1.axhline(0, color='grey', linestyle=':', linewidth=0.8)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.set_title('Cumulative Return Based on Optimal Strategy')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(dates, cum_reward_arr[1:], 'r-', linewidth=1.5)
        ax2.axhline(0, color='grey', linestyle=':', linewidth=0.8)
        ax2.set_ylabel('Cumulative Reward (%)')
        ax2.set_xlabel('Date')
        ax2.set_title('Cumulative Reward (Sum of Stepwise Portfolio Returns)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('cumulative_reward.png', dpi=150)
        plt.show()

        print(f"\n========== 历史回测统计 ==========")
        print(f"回测天数:     {n_days}")
        print(f"初始财富:     {W0:.2f}")
        print(f"终端财富:     {W:.4f}")
        print(f"累计收益:     {cum_return_pct[-1]:.2f}%")
        print(f"累计奖励:     {cum_reward_arr[-1]:.2f}%")


# ========== 使用示例 ==========

# 1. 准备数据（示例：生成模拟价格数据）
dates = pd.date_range('2020-01-01', periods=500, freq='D')
n_assets = 3
price_data = pd.DataFrame(
    np.cumprod(1 + np.random.randn(500, n_assets) * 0.01, axis=0),
    index=dates,
    columns=[f'Asset_{i}' for i in range(n_assets)]
)

# 2. 初始化并训练
fvi = FVIWithData(price_data, cash_rate=0.03, T=10)
fvi.train(n_states=200, n_actions=100, n_samples=50)

# 3. 获取最优动作
W = 1.0  # 当前财富
p = np.array([0.3, 0.3, 0.3])  # 当前仓位
t = 0
action = fvi.get_action(W, p, t)
print(f"最优动作: {action}")
print(f"新仓位: {action / action.sum()}")

# 4. 在历史数据上看累计收益
print("\n开始历史回测...")
fvi.backtest_cumulative(W0=W, p0=p)