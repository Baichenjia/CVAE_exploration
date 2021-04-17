import tensorflow as tf
import tensorflow_probability as tfp
import os
import numpy as np
tfd = tfp.distributions


def standard_normal(params):
    n = params.shape[0]
    d = params.shape[-1]                       # channel
    mu = tf.zeros_like(params[..., :d // 2])   # 均值为0
    sigma = tf.ones_like(params[..., d // 2:])
    standard_distr = tfd.Normal(loc=tf.stop_gradient(mu), scale=tf.stop_gradient(sigma))  # 高斯
    return standard_distr


def normal_parse_params(params, min_sigma=0.0):
    """
    将输入拆分成两份, 分别代表 mean 和 std.
    min_sigma 是对 sigma 最小值的限制
    """
    n = params.shape[0]
    d = params.shape[-1]                    # channel
    mu = params[..., :d // 2]               # 最后一维的通道分成两份, 分别为均值和标准差
    sigma_params = params[..., d // 2:]
    sigma = tf.math.softplus(sigma_params)
    sigma = tf.clip_by_value(t=sigma, clip_value_min=min_sigma, clip_value_max=1e5)

    distr = tfd.Normal(loc=mu, scale=sigma)   # 高斯
    # proposal 网络的输出 (None,None,256), mu.shape=(None,None,128), sigma.shape=(None,None,128)
    return distr


def rec_log_prob(rec_params, s_next, min_sigma=1e-2):
    # rec_params.shape = (None, None, 1024), 前一半参数代表均值, 后一半参数代表标准差.
    distr = normal_parse_params(rec_params, min_sigma)
    log_prob = distr.log_prob(s_next)               # (None, None, 512)
    assert len(log_prob.get_shape().as_list()) == 3 and log_prob.get_shape().as_list()[-1] == 512
    return tf.reduce_sum(log_prob, axis=-1)

