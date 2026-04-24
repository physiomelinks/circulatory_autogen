import numpy as np
import os
import sys
import sympy

"""
These functions can be used as cost functions. Specify a name of one of these functions as the "cost_type" in obs_data.json to
use it as the cost.

When making your own cost function make sure it works for scalars and vectors. Otherwise put an error message so that if it is used
for the wrong data type it gets called out and stopped.
"""
def is_MLE(func):
    func.is_MLE = True
    return func

@is_MLE
def gaussian_MLE(output, desired_mean, std, weight):
    # cost = np.sum(np.power(updated_weight_const_vec*(const -
    #                            self.obs_info["ground_truth_const"])/self.obs_info["std_const_vec"], 2))

    # cost = series_cost = np.sum(np.power((series[:, :min_len_series] -
    #                                            self.obs_info["ground_truth_series"][:,
    #                                            :min_len_series]) * updated_weight_series_vec.reshape(-1, 1) /
    #                                           self.obs_info["std_series_vec"].reshape(-1, 1), 2)) / min_len_series

    cost = np.power((output - desired_mean)/std, 2)*weight
    if hasattr(output, '__len__'):
        # if entry is a vector then turn the vector of costs for each data point into a average cost
        cost = np.sum(cost)/len(output)
    
    return cost

# TODO we need to create derivative functions for each cost with respect to the outputs so that we can pass 

def MSE(*args, **kwargs):
    # The mean squared error cost is the same as the 
    return gaussian_MLE(*args, **kwargs)

@is_MLE
def multimodal_gaussian(output, prob_dist_params, weight):
    if hasattr(output, '__len__'):
        print("ERROR: multimodal_gaussian cost function is not implemented for series data")
        # if entry is a vector then turn the vector of costs for each data point into a average cost
        # cost_mode = np.sum(cost_mode)/len(output)
    
    # TODO make it so the below checks are only done once.
    # TODO checks should be done when parsing rather than every time this is called.
    allowable_keys_list = ["means", "stds", "scales"]
    allowable_keys_list.sort()
    keys_list = [*prob_dist_params]
    keys_list.sort()
    if not isinstance(prob_dist_params, dict):
        print('!!!!!!!!!!!!')
        print("ERROR prob_dist_params in obs_data.json needs to be a dict! The entries should be:")
        print(allowable_keys_list)
        print('!!!!!!!!!!!!')
        exit()

    if keys_list != allowable_keys_list:
        print('!!!!!!!!!!!!')
        print("ERROR prob_dist_params in obs_data.json needs to be a dict with entries:")
        print(allowable_keys_list)
        print('!!!!!!!!!!!!')
        exit()

    if sum(prob_dist_params["scales"]) != 1:
        print('!!!!!!!!!!!!')
        print("ERROR scales in prob_dist_params for multimodal_gaussian in obs_data.json need to sum to 1")
        print('!!!!!!!!!!!!')
        exit()

    # apply log-sum-exp trick to avoid numerical instability with large exp values
    # this is log(sum(exp(v_i))) = max(v) + log(sum_i(exp(v_i - max(v))) 

    v_vec = np.zeros(len(prob_dist_params["means"]))
    for idx, (desired_mean, std, scale) in enumerate(zip(prob_dist_params["means"], prob_dist_params["stds"], prob_dist_params["scales"])):
        v_vec[idx] = np.power((output - desired_mean)/std, 2)*scale
    
    v_max = np.max(v_vec)
    sum_inner_term = np.sum(np.exp(v_vec - v_max))

    cost = (v_max + np.log(sum_inner_term))*weight
            
    
    # # below was before applying more efficient log-sum-exp trick. 
    # cost_exp = 0
    # for idx, (desired_mean, std, scale) in enumerate(zip(prob_dist_params["means"], prob_dist_params["stds"], prob_dist_params["scales"])):
    #     cost_mode = np.power((output - desired_mean)/std, 2)*scale
    #     cost_exp += np.exp(cost_mode)
    
    # cost_check = np.log(cost_exp)*weight

    # print(cost)
    # print(cost_check)

    return cost

def AE(output, desired_mean, std, weight):
    cost = np.abs((output - desired_mean)/std)*weight
    if hasattr(output, '__len__'):
        # if entry is a vector then turn the vector of costs for each data point into a average cost
        cost = np.sum(cost)/len(output)

# decorator for cost combiners
def cost_combiner(func):
    func.cost_combiner = True
    return func

@is_MLE
@cost_combiner
def additive(costs):
    cost = np.sum(costs)
    return cost

@cost_combiner
def norm_additive(costs):
    cost = np.sum(costs)/len(costs)
    return cost








