"""
Ռասիստական անիկդոտվա պետք սկսել fairness ի paper ի իմպլեմենտացիան

Միհատ մարդ փողոցով քայլումա, մեկել տենումա միհատ նեգռ, ձեռնել միհատ մեծ էկռանով 
թազա առած տելեվիզռ։ Տենց մտքներովա ընգնում "Ախպեր կարողա իմնա, իմնելա մոտավոր էս չափի"
Տենց միքիչելա տատանվում, վերջը գնումա տուն ստուգի։ Հասնումա տուն պարզվումա չէ ՝
իրանը բասսեիննա լվանում
"""

# date finished -> 20.08.2020

import numpy as np
import pandas as pd
import scipy.optimize as optim
from sklearn.preprocessing import StandardScaler

from implem_helpers import *

# data is preprocessed just like in paper, using command pd.get_dummies()
data = pd.read_csv('german.csv')

AGE_SENSITIVE = 28  # Since I wasn't able to find out how paper's authors 
                    # seperated into sensitive and non_sensitive classes 
                    # I'm using number that seperates qchic shatic evenly


# seperation into sensitive and non sensitive
data_sensitive = data[data.age > AGE_SENSITIVE]
data_non_sensitive = data[data.age <= AGE_SENSITIVE]
y_sensitive = data_sensitive.default
y_non_sensitive = data_non_sensitive.default

print (f'Dataset contains {data.shape[0]} examples and {data.shape[1]} features')
print (f'From which {data_sensitive.shape[0]} belong to sensitive group and {data_non_sensitive.shape[0]} to non nensitive group ')

del data_sensitive['default']
del data_non_sensitive['default']

# Standard Scaling
data_sensitive = StandardScaler().fit_transform(data_sensitive)
data_non_sensitive = StandardScaler().fit_transform(data_non_sensitive)


NUM_PROTOTYPES = 10
num_features = data_sensitive.shape[1]

params = np.random.uniform(size=(num_features * 2 + NUM_PROTOTYPES + NUM_PROTOTYPES * num_features))
# here we generate random weight for each of the features both for sensitive data
# and for non sensitive, hence num_features*2(in paper this is denoted as alpha)
# alphas are used for calculating distances

# Then NUM_PROTOTYPES is a weight for each prototype, this is multiplied with 
# M_nk s and used for calculating y_hat

# Next is NUM_PROTOTYPES * num_features which is v(in paper), this is also used
# for calculating distances


bnd = [] # This is needed for l-bfgs algorithm
for i, _ in enumerate(params):
    if i < num_features * 2 or i >= num_features * 2 + NUM_PROTOTYPES:
        bnd.append((None, None))
    else:
        bnd.append((0, 1))

new_params = optim.fmin_l_bfgs_b(optim_objective, x0=params, epsilon=1e-5,
                                  args=(data_sensitive, data_non_sensitive,
                                        y_sensitive, y_non_sensitive),
                                  bounds=bnd, approx_grad=True, maxfun=10_000,
                                  maxiter=10_000)[0]


x_hat_senitive, x_hat_nons, y_hat_sens, y_hat_nons = optim_objective(new_params,data_sensitive, data_non_sensitive,
                                        y_sensitive, y_non_sensitive, inference=True)

print ('Done')
