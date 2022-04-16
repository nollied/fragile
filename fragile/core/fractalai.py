from typing import Callable

import judo
from judo.data_types import dtype
from judo.functions.random import random_state
from judo.judo_tensor import tensor
from judo.typing import Tensor
import numpy


AVAILABLE_FUNCTIONS = {
    "l2_norm",
    "relativize",
    "get_alive_indexes",
    "calculate_virtual_reward",
    "calculate_clone",
    "calculate_distance",
    "fai_iteration",
    "cross_virtual_reward",
    "cross_clone",
    "cross_fai_iteration",
}


def l2_norm(x: Tensor, y: Tensor) -> Tensor:
    """Euclidean distance between two batches of points stacked across the first dimension."""
    return judo.norm(x - y, axis=1)


def relativize(x: Tensor) -> Tensor:
    """Normalize the data using a custom smoothing technique."""
    orig = x
    x = judo.astype(x, dtype.float)
    std = x.std()
    if float(std) == 0:
        return judo.ones(len(x), dtype=orig.dtype)
    standard = (x - x.mean()) / std
    with numpy.errstate(invalid="ignore", divide="ignore"):
        res = judo.where(standard > 0.0, judo.log(1.0 + standard) + 1.0, judo.exp(standard))
    return res


def get_alive_indexes(oobs: Tensor):
    """Get indexes representing random alive walkers given a vector of death conditions."""
    if judo.all(oobs):
        return judo.arange(len(oobs))
    ix = judo.logical_not(oobs).flatten()
    return random_state.choice(judo.arange(len(ix))[ix], size=len(ix), replace=ix.sum() < len(ix))


def calculate_distance(
    observs: Tensor,
    distance_function: Callable = l2_norm,
    return_compas: bool = False,
    oobs: Tensor = None,
    compas: Tensor = None,
):
    """Calculate a distance metric for each walker with respect to a random companion."""
    if compas is None:
        compas = get_alive_indexes(oobs) if oobs is not None else judo.arange(observs.shape[0])
        compas = random_state.permutation(compas)
    flattened_observs = observs.view(observs.shape[0], -1)
    distance = distance_function(flattened_observs, flattened_observs[compas])
    distance_norm = relativize(distance.flatten())
    return distance_norm if not return_compas else (distance_norm, compas)


def calculate_virtual_reward(
    observs: Tensor,
    rewards: Tensor,
    oobs: Tensor = None,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    other_reward: Tensor = 1.0,
    return_compas: bool = False,
    distance_function: Callable = l2_norm,
):
    """Calculate the virtual rewards given the required data."""

    compas = get_alive_indexes(oobs) if oobs is not None else judo.arange(len(rewards))
    compas = random_state.permutation(compas)
    flattened_observs = observs.reshape(len(oobs), -1)
    other_reward = other_reward.flatten() if dtype.is_tensor(other_reward) else other_reward
    distance = distance_function(flattened_observs, flattened_observs[compas])
    distance_norm = relativize(distance.flatten())
    rewards_norm = relativize(rewards.flatten())
    virtual_reward = distance_norm**dist_coef * rewards_norm**reward_coef * other_reward
    return virtual_reward if not return_compas else (virtual_reward, compas)


def calculate_clone(virtual_rewards: Tensor, oobs: Tensor = None, eps=1e-3):
    """Calculate the clone indexes and masks from the virtual rewards."""
    compas_ix = get_alive_indexes(oobs) if oobs is not None else judo.arange(len(virtual_rewards))
    compas_ix = random_state.permutation(compas_ix)
    vir_rew = virtual_rewards.flatten()
    clone_probs = (vir_rew[compas_ix] - vir_rew) / judo.where(vir_rew > eps, vir_rew, tensor(eps))
    will_clone = clone_probs.flatten() > random_state.random(len(clone_probs))
    return compas_ix, will_clone


def fai_iteration(
    observs: Tensor,
    rewards: Tensor,
    oobs: Tensor = None,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    eps=1e-8,
    other_reward: Tensor = 1.0,
):
    """Perform a FAI iteration."""
    oobs = oobs if oobs is not None else judo.zeros(rewards.shape, dtype=dtype.bool)
    virtual_reward = calculate_virtual_reward(
        observs,
        rewards,
        oobs,
        dist_coef=dist_coef,
        reward_coef=reward_coef,
        other_reward=other_reward,
    )
    compas_ix, will_clone = calculate_clone(virtual_rewards=virtual_reward, oobs=oobs, eps=eps)
    return compas_ix, will_clone


def cross_virtual_reward(
    host_observs: Tensor,
    host_rewards: Tensor,
    ext_observs: Tensor,
    ext_rewards: Tensor,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    return_compas: bool = False,
    distance_function: Callable = l2_norm,
):
    """Calculate the virtual rewards between two cloud of points."""
    host_observs = host_observs.reshape(len(host_rewards), -1)
    ext_observs = ext_observs.reshape(len(ext_rewards), -1)
    compas_host = random_state.permutation(judo.arange(len(host_rewards)))
    compas_ext = random_state.permutation(judo.arange(len(ext_rewards)))

    # TODO: check if it's better for the distances to be the same for host and ext
    h_dist = distance_function(host_observs, ext_observs[compas_host])
    e_dist = distance_function(ext_observs, host_observs[compas_ext])
    host_distance = relativize(h_dist.flatten())
    ext_distance = relativize(e_dist.flatten())

    host_rewards = relativize(host_rewards)
    ext_rewards = relativize(ext_rewards)

    host_vr = host_distance**dist_coef * host_rewards**reward_coef
    ext_vr = ext_distance**dist_coef * ext_rewards**reward_coef
    if return_compas:
        return (host_vr, compas_host), (ext_vr, compas_ext)
    return host_vr, ext_vr


def cross_clone(
    host_virtual_rewards: Tensor,
    ext_virtual_rewards: Tensor,
    host_oobs: Tensor = None,
    eps=1e-3,
):
    """Perform a clone operation between two different groups of points."""
    compas_ix = random_state.permutation(judo.arange(len(ext_virtual_rewards)))
    host_vr = judo.astype(host_virtual_rewards.flatten(), dtype=dtype.float32)
    ext_vr = judo.astype(ext_virtual_rewards.flatten(), dtype=dtype.float32)
    clone_probs = (ext_vr[compas_ix] - host_vr) / judo.where(
        ext_vr > eps,
        ext_vr,
        tensor(eps, dtype=dtype.float32),
    )
    will_clone = clone_probs.flatten() > random_state.random(len(clone_probs))
    if host_oobs is not None:
        will_clone[host_oobs] = True
    return compas_ix, will_clone


def cross_fai_iteration(
    host_observs: Tensor,
    host_rewards: Tensor,
    ext_observs: Tensor,
    ext_rewards: Tensor,
    host_oobs: Tensor = None,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    distance_function: Callable = l2_norm,
    eps: float = 1e-8,
):
    """Perform a FractalAI cloning process between two clouds of points."""
    host_vr, ext_vr = cross_virtual_reward(
        host_observs=host_observs,
        host_rewards=host_rewards,
        ext_observs=ext_observs,
        ext_rewards=ext_rewards,
        dist_coef=dist_coef,
        reward_coef=reward_coef,
        distance_function=distance_function,
        return_compas=False,
    )

    compas_ix, will_clone = cross_clone(
        host_virtual_rewards=host_vr,
        ext_virtual_rewards=ext_vr,
        host_oobs=host_oobs,
        eps=eps,
    )
    return compas_ix, will_clone
