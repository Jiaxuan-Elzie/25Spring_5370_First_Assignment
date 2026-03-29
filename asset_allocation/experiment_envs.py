import numpy as np

from .common import cara_utility


class SingleAdvantageEnv:
    def __init__(
        self,
        n_risky=3,
        mu_good=0.12,
        mu_bad=0.03,
        sigma_good=0.15,
        sigma_bad=0.15,
        cash_rate=0.02,
        horizon=6,
        init_weights=None,
        init_wealth=1.0,
        risk_aversion=1.5,
        seed=None,
    ):
        self.n_risky = n_risky
        self.n_assets_total = n_risky + 1
        self.mu_good = mu_good
        self.mu_bad = mu_bad
        self.sigma_good = sigma_good
        self.sigma_bad = sigma_bad
        self.cash_rate = cash_rate
        self.T = horizon
        self.init_wealth = init_wealth
        self.risk_aversion = risk_aversion
        self.rng = np.random.default_rng(seed)

        if init_weights is None:
            init_weights = np.ones(self.n_assets_total) / self.n_assets_total
        self.init_weights = np.array(init_weights, dtype=float)

        assert len(self.init_weights) == self.n_assets_total
        assert np.isclose(np.sum(self.init_weights), 1.0)
        assert np.all(self.init_weights >= 0)

        self.means = np.array([self.mu_good] + [self.mu_bad] * (self.n_risky - 1), dtype=float)
        self.stds = np.array([self.sigma_good] + [self.sigma_bad] * (self.n_risky - 1), dtype=float)

        self.reset()

    def reset(self):
        self.t = 0
        self.wealth = self.init_wealth
        self.weights = self.init_weights.copy()
        return self._state()

    def _state(self):
        return np.concatenate(([self.t / self.T, np.log(max(self.wealth, 1e-8))], self.weights))

    def step(self, new_weights):
        new_weights = np.asarray(new_weights, dtype=float)
        old_weights = self.weights.copy()

        risky_returns = self.rng.normal(self.means, self.stds)
        gross_returns = np.concatenate(([1.0 + self.cash_rate], 1.0 + risky_returns))

        portfolio_gross = np.dot(new_weights, gross_returns)
        next_wealth = self.wealth * portfolio_gross

        next_values = self.wealth * new_weights * gross_returns
        next_weights = next_values / np.sum(next_values)

        self.wealth = next_wealth
        self.weights = next_weights
        self.t += 1

        done = self.t == self.T
        reward = cara_utility(self.wealth, self.risk_aversion) if done else 0.0
        turnover = np.sum(np.abs(new_weights - old_weights))

        return self._state(), reward, done, {
            "wealth": self.wealth,
            "weights": self.weights.copy(),
            "executed_weights": new_weights.copy(),
            "risky_returns": risky_returns,
            "gross_returns": gross_returns,
            "turnover": turnover,
            "regime": 0,
        }


class TwoStateRotationEnv:
    def __init__(
        self,
        n_risky=3,
        mu_hi=0.15,
        mu_lo=-0.02,
        sigma=0.12,
        cash_rate=0.02,
        horizon=6,
        stay_prob=0.9,
        init_weights=None,
        init_wealth=1.0,
        risk_aversion=1.5,
        seed=None,
    ):
        self.n_risky = n_risky
        self.n_assets_total = self.n_risky + 1
        self.mu_hi = mu_hi
        self.mu_lo = mu_lo
        self.sigma = sigma
        self.cash_rate = cash_rate
        self.T = horizon
        self.stay_prob = stay_prob
        self.init_wealth = init_wealth
        self.risk_aversion = risk_aversion
        self.rng = np.random.default_rng(seed)

        if init_weights is None:
            init_weights = np.ones(self.n_assets_total) / self.n_assets_total
        self.init_weights = np.array(init_weights, dtype=float)

        assert len(self.init_weights) == self.n_assets_total
        assert np.isclose(np.sum(self.init_weights), 1.0)
        assert np.all(self.init_weights >= 0)

        self.reset()

    def _sample_regime_path(self):
        regimes = np.zeros(self.T, dtype=int)
        regimes[0] = self.rng.integers(0, 2)
        for time_index in range(1, self.T):
            if self.rng.random() < self.stay_prob:
                regimes[time_index] = regimes[time_index - 1]
            else:
                regimes[time_index] = 1 - regimes[time_index - 1]
        return regimes

    def reset(self):
        self.t = 0
        self.wealth = self.init_wealth
        self.weights = self.init_weights.copy()
        self.regimes = self._sample_regime_path()
        return self._state()

    def _regime_one_hot(self, regime):
        if regime == 0:
            return np.array([1.0, 0.0], dtype=float)
        return np.array([0.0, 1.0], dtype=float)

    def _state(self):
        regime = self.regimes[self.t]
        return np.concatenate((
            [self.t / self.T, np.log(max(self.wealth, 1e-8))],
            self.weights,
            self._regime_one_hot(regime),
        ))

    def _current_means(self):
        regime = self.regimes[self.t]
        means = np.zeros(self.n_risky, dtype=float)

        if regime == 0:
            means[0] = self.mu_hi
            if self.n_risky > 1:
                means[1] = self.mu_lo
        else:
            means[0] = self.mu_lo
            if self.n_risky > 1:
                means[1] = self.mu_hi

        return means

    def step(self, new_weights):
        new_weights = np.asarray(new_weights, dtype=float)
        old_weights = self.weights.copy()
        regime = self.regimes[self.t]

        means = self._current_means()
        risky_returns = self.rng.normal(means, self.sigma, size=self.n_risky)
        gross_returns = np.concatenate(([1.0 + self.cash_rate], 1.0 + risky_returns))

        portfolio_gross = np.dot(new_weights, gross_returns)
        next_wealth = self.wealth * portfolio_gross

        next_values = self.wealth * new_weights * gross_returns
        next_weights = next_values / np.sum(next_values)

        self.wealth = next_wealth
        self.weights = next_weights
        self.t += 1

        done = self.t == self.T
        reward = cara_utility(self.wealth, self.risk_aversion) if done else 0.0
        turnover = np.sum(np.abs(new_weights - old_weights))

        if done:
            next_state = np.concatenate((
                [1.0, np.log(max(self.wealth, 1e-8))],
                self.weights,
                np.array([0.0, 0.0], dtype=float),
            ))
        else:
            next_state = self._state()

        return next_state, reward, done, {
            "wealth": self.wealth,
            "weights": self.weights.copy(),
            "executed_weights": new_weights.copy(),
            "risky_returns": risky_returns,
            "gross_returns": gross_returns,
            "turnover": turnover,
            "regime": regime,
        }
