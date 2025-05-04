import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from toolz import partial
from scipy.optimize import fmin_slsqp

class SyntheticControl:
    def __init__(self, data, metric, period_index, shopno, treated, after_treatment, bootstrap_rounds=100, seed=42):
        self.data = data.copy()
        self.metric = metric
        self.period_index = period_index
        self.shopno = shopno
        self.treated = treated
        self.after_treatment = after_treatment
        self.bootstrap_rounds = bootstrap_rounds
        self.seed = seed

    def make_random_placebo(self):
        control = self.data.query(f"~{self.treated}")
        shopnos = control[self.shopno].unique()
        placebo_shopno = np.random.choice(shopnos)
        return control.assign(**{self.treated: control[self.shopno] == placebo_shopno})
    
    def loss(self, W, X, y):
        return np.sqrt(np.mean((y - X.dot(W))**2))

    def synthetic_control(self, data=None):
        if data is None:
            data = self.data

        df_pre_control = (data
            .query(f"not {self.treated}")
            .query(f"not {self.after_treatment}")
            .pivot(index=self.period_index,
                   columns=self.shopno,
                   values=self.metric)
        )

        self.control_units_ = list(df_pre_control.columns)

        self.X = df_pre_control.values

        self.y = (data
                            .query(f"not {self.after_treatment}")
                            .query(f"{self.treated}")
                            .groupby(self.period_index)[self.metric]
                            .mean())
        
        self.n_features = self.X.shape[1]
        self.init_w = np.ones(self.n_features) / self.n_features
        
        cons = lambda w: np.sum(w) - 1
        bounds = [(0.0, 1.0)] * self.n_features

        opt_w = fmin_slsqp(
            partial(self.loss, X=self.X, y=self.y),
            self.init_w,
            f_eqcons=cons,
            bounds=bounds,
            disp=False
        )

        self.w_ = opt_w 


        x_all_control = (data
                         .query(f"not {self.treated}")
                         .pivot(index=self.period_index, columns=self.shopno, values=self.metric)
                         .values)


        sc_series = x_all_control @ opt_w

        y_post_treat = data.query(f"{self.treated} and {self.after_treatment}")[self.metric].values
        sc_post = sc_series[-len(y_post_treat):]
        att = np.mean(y_post_treat - sc_post)
        return att, opt_w
    
    def estimate_se_sc(self, alpha=0.05):
        np.random.seed(self.seed)
        att, _ = self.synthetic_control()

        effects = []
        for _ in range(self.bootstrap_rounds):
            placebo_data = self.make_random_placebo()
            # опять же распаковать только att_placebo
            att_placebo, _ = self.synthetic_control(data=placebo_data)
            effects.append(att_placebo)

        # теперь effects — просто список чисел
        se = np.std(effects, ddof=1)

        z = norm.ppf(1 - alpha / 2)
        ci_lower = att - z * se
        ci_upper = att + z * se

        return se, ci_lower, ci_upper

