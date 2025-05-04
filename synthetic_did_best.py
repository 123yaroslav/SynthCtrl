import numpy as np
import pandas as pd
import cvxpy as cp
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import fmin_slsqp
from functools import partial

class SyntheticDIDModel:
    def __init__(self, data, metric, period_index, shopno, treated, after_treatment,
                 seed=42, bootstrap_rounds=100, njobs=4):
        self.data = data.copy()
        self.outcome_col = metric
        self.period_index_col = period_index
        self.shopno_col = shopno
        self.treat_col = treated
        self.post_col = after_treatment
        self.seed = seed
        self.bootstrap_rounds = bootstrap_rounds
        self.njobs = njobs

    def loss(self, w, X, y):
        pred = X.T.dot(w)
        return np.sqrt(np.mean((y - pred)**2))
    
    def loss_penalized(self, w, X, y, T_pre, zeta):
        resid = X.dot(w) - y
        return np.sum(resid**2) + T_pre * (zeta**2) * np.sum(w[1:]**2)

    def calculate_regularization(self, data):
        if self.post_col not in data.columns or self.treat_col not in data.columns:
            raise ValueError(f"Отсутствуют необходимые столбцы: {self.post_col} или {self.treat_col}")
            
        n_treated_post = data.loc[(data[self.post_col] == 1) & (data[self.treat_col] == 1)].shape[0]
        first_diff_std = (data
                          .loc[(data[self.post_col] == 0) & (data[self.treat_col] == 0)]
                          .sort_values(self.period_index_col)
                          .groupby(self.shopno_col)[self.outcome_col]
                          .diff()
                          .std())
        return n_treated_post ** (1 / 4) * first_diff_std

    def join_weights(self, data, unit_w, time_w):
        joined = (data
                  .set_index([self.period_index_col, self.shopno_col])
                  .join(time_w)
                  .join(unit_w)
                  .reset_index()
                  .fillna({
                      time_w.name: 1 / len(pd.unique(data.loc[data[self.post_col] == 1, self.period_index_col])),
                      unit_w.name: 1 / len(pd.unique(data.loc[data[self.treat_col] == 1, self.shopno_col]))
                  })
                  .assign(**{"weights": lambda d: (d[time_w.name] * d[unit_w.name]).round(10)})
                  .astype({self.treat_col: int, self.post_col: int}))
        return joined

    def fit_time_weights(self, data):
        control = data.loc[data[self.treat_col] == 0]
        y_pre = (control
                 .loc[control[self.post_col] == 0]
                 .pivot(index=self.period_index_col, columns=self.shopno_col, values=self.outcome_col))
        y_post_mean = (control
                       .loc[control[self.post_col] == 1]
                       .groupby(self.shopno_col)[self.outcome_col]
                       .mean()
                       .values)

        X = np.vstack([np.ones((1, y_pre.shape[1])), y_pre.values])
        n_features, n_shops = X.shape
        init_w = np.ones(n_features) / n_features

        cons = lambda w, *args: np.sum(w[1:]) - 1

        bounds = [(None, None)] + [(0.0, 1.0)] * (n_features - 1)

        opt_w = fmin_slsqp(
            func=partial(self.loss, X=X, y=y_post_mean),
            x0=init_w,
            f_eqcons=cons,
            bounds=bounds,
            disp=False
        )

        return pd.Series(opt_w[1:], name="time_weights", index=y_pre.index)

    def fit_unit_weights(self, data):
        zeta = self.calculate_regularization(data)
        pre_data = data.loc[data[self.post_col] == 0]
        y_pre_control = (pre_data
                         .loc[pre_data[self.treat_col] == 0]
                         .pivot(index=self.period_index_col, columns=self.shopno_col, values=self.outcome_col))
        y_pre_treat_mean = (pre_data
                            .loc[pre_data[self.treat_col] == 1]
                            .groupby(self.period_index_col)[self.outcome_col]
                            .mean())
        T_pre = y_pre_control.shape[0]
        
        X = np.concatenate([np.ones((T_pre, 1)), y_pre_control.values], axis=1)
        
        cons = lambda w, *args: np.sum(w[1:]) - 1

        n_coef = X.shape[1]
        init_w = np.ones(n_coef) / n_coef

        bounds = [(None, None)] + [(0.0, 1.0)] * (n_coef - 1)
        

        opt_w = fmin_slsqp(
            func = partial(self.loss_penalized, X=X, y=y_pre_treat_mean.values, T_pre=T_pre, zeta=zeta),
            x0 = init_w,
            f_eqcons = cons,
            bounds = bounds,
            disp = False
        )
        return pd.Series(opt_w[1:], name="unit_weights", index=y_pre_control.columns), opt_w[0]

    def synthetic_diff_in_diff(self, data=None):
        if data is None:
            data = self.data
        unit_weights, intercept = self.fit_unit_weights(data)
        time_weights = self.fit_time_weights(data)
        did_data = self.join_weights(data, unit_weights, time_weights)
        formula = f"{self.outcome_col} ~ {self.post_col}*{self.treat_col}"
        did_model = smf.wls(formula, data=did_data, weights=did_data["weights"] + 1e-10).fit()
        att = did_model.params[f"{self.post_col}:{self.treat_col}"]
        return att, unit_weights, time_weights, did_model, intercept

    def make_random_placebo(self, data):
        control = data.query(f"~{self.treat_col}")
        shopnos = control[self.shopno_col].unique()
        placebo_shopno = np.random.choice(shopnos)
        return control.assign(**{self.treat_col: control[self.shopno_col] == placebo_shopno})


    def _single_placebo_att(self, seed):
        np.random.seed(seed)
        placebo_data = self.make_random_placebo(self.data)
        att_placebo, *_ = self.synthetic_diff_in_diff(data=placebo_data)
        return att_placebo

    def estimate_se(self, alpha=0.05):
        master_rng = np.random.RandomState(self.seed)
        main_att, *_ = self.synthetic_diff_in_diff()

        seeds = master_rng.randint(low=0, high=2**31-1,
                                   size=self.bootstrap_rounds)

        effects = Parallel(n_jobs=self.njobs)(
            delayed(self._single_placebo_att)(seed)
            for seed in seeds
        )

        se = np.std(effects, ddof=1)
        z  = norm.ppf(1 - alpha/2)
        return main_att, se, main_att - z*se, main_att + z*se
    
