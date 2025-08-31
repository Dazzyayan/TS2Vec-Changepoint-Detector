import numpy as np

# 
# This file generates simulated time series data with changepoints in mean, variance, and both.
# The data is generated from the normal distribution.
# Those labeled as _data return i.i.d. data.
# Those labelled as _ar_data return AR(1) data.
# 


def _generate_from_distribution(dist_type, size, params=None):
    """Helper function to generate data from different distributions."""
    if params is None:
        params = {}
    
    if dist_type == 'normal':
        loc = params.get('loc', 0)
        scale = params.get('scale', 1)
        return np.random.normal(loc, scale, size)
    elif dist_type == 'uniform':
        low = params.get('low', -1)
        high = params.get('high', 1)
        return np.random.uniform(low, high, size)


def generate_changepoint_data(sigma_i=1, mu_1=0, nu=50, buffer_period=100, M=5000, 
                             S=[0.25, 0.5, 1, 3], theta_values=[1, -1], 
                             starting_dist='normal', starting_params=None):
    """
    This simulated data generation method is based on 
    [1] D. A. Bodenham and N. M. Adams, 'Continuous monitoring for changepoints in data streams using adaptive estimation',
    Stat Comput, vol. 27, no. 5, pp. 1257–1270, Sep. 2017, doi: 10.1007/s11222-016-9684-8.
    
    This is an alternative way to generate a time series with changepoints in mean. It's inclusion of a grace period and sufficient time for the model to detect the changepoint means
    it gives a simulated scenario of an appropriate level of difficulty.
    We replace grace period with buffer period.
    
    Args:
        starting_dist: Distribution type for initial data ('normal')
        starting_params: Dict with distribution parameters. If None, uses defaults based on starting_dist
    """
    
    # Set default parameters for starting distribution
    if starting_params is None:
        if starting_dist == 'normal':
            starting_params = {'loc': mu_1, 'scale': sigma_i}
    
    # Handle M=0 case: generate time series with no changepoints
    if M == 0:
        total_length = buffer_period + np.random.poisson(nu)
        timeseries = _generate_from_distribution(starting_dist, total_length, starting_params)
        parameters = np.full((total_length, 1), mu_1)
        return timeseries, [], parameters
    
    # Generate changepoints: tau_i = tau_{i-1} + buffer_period + xi_i
    changepoints = []
    for i in range(M):
        xi_i = np.random.poisson(nu) # random interval
        if i == 0:
            tau_i = buffer_period + xi_i # buffer period + random interval
        else:
            tau_i = changepoints[i-1] + buffer_period + xi_i # changepoint + buffer period + random interval
        changepoints.append(tau_i)
    
    # Generate means: mu_i = mu_{i-1} + theta * delta_i * sigma_i (S measured in standard deviations)
    means = [mu_1]
    for i in range(1, M):
        delta_i = np.random.choice(S) # choose a random increment in standard deviations
        theta = np.random.choice(theta_values) # theta allows us to choose the direction of the change
        mu_i = means[i-1] + theta * delta_i * sigma_i # increment is added to the previous mean
        means.append(mu_i)
    
    # Generate the timeseries data
    total_length = changepoints[-1] + buffer_period + np.random.poisson(nu)
    timeseries = np.zeros(total_length)
    parameters = np.zeros(total_length)
    
    # Generate data for each segment
    # Segment 0: from start to first changepoint (use starting distribution)
    if changepoints[0] > 0:
        timeseries[0:changepoints[0]] = _generate_from_distribution(starting_dist, changepoints[0], starting_params)
        parameters[0:changepoints[0]] = mu_1
    
    # Segments 1 to M: from each changepoint to the next (use normal distribution with changing means)
    for i in range(M):
        start_idx = changepoints[i]
        end_idx = changepoints[i + 1] if i + 1 < len(changepoints) else total_length
        
        if start_idx < end_idx:
            segment_length = end_idx - start_idx
            timeseries[start_idx:end_idx] = np.random.normal(means[i], sigma_i, segment_length)
            parameters[start_idx:end_idx] = means[i]
    
    parameters = parameters.reshape(-1, 1)
    
    return timeseries, changepoints, parameters



def generate_changepoint_ar_data(sigma_i=1, mu_1=0, nu=50, buffer_period=100, M=5000, 
                                S=[0.25, 0.5, 1, 3], theta_values=[1, -1], phi=0.7,
                                starting_dist='normal', starting_params=None):
    """Generate AR(1) time series with changepoints in mean.
    """
    
    # Set default parameters for starting distribution
    if starting_params is None:
        if starting_dist == 'normal':
            starting_params = {'loc': mu_1, 'scale': sigma_i}
            
    # Handle M=0 case: generate AR(1) time series with no changepoints
    if M == 0:
        total_length = buffer_period + np.random.poisson(nu)
        timeseries = np.zeros(total_length)
        parameters = np.full((total_length, 1), mu_1)
        
        timeseries[0] = _generate_from_distribution(starting_dist, 1, starting_params)[0]
        for t in range(1, total_length):
            timeseries[t] = mu_1 + phi * (timeseries[t-1] - mu_1) + np.random.normal(0, sigma_i)
        
        return timeseries, [], parameters
    
    # Generate changepoints. Using the same mean generation method as the previous function
    # Future refactor to move mean generation to a separate function
    changepoints = []
    for i in range(M):
        xi = np.random.poisson(nu)
        tau = buffer_period + xi if i == 0 else changepoints[-1] + buffer_period + xi
        changepoints.append(tau)
    
    # Generate means
    means = [mu_1]
    for i in range(1, M):
        delta = np.random.choice(S)
        theta = np.random.choice(theta_values)
        means.append(means[-1] + theta * delta * sigma_i)
    
    # Generate time series
    total_length = changepoints[-1] + buffer_period + np.random.poisson(nu)
    timeseries = np.zeros(total_length)
    parameters = np.zeros(total_length)
    
    timeseries[0] = _generate_from_distribution(starting_dist, 1, starting_params)[0]
    parameters[0] = mu_1
    
    mean_idx = 0
    for t in range(1, total_length):
        if mean_idx < len(changepoints) and t > changepoints[mean_idx]:
            mean_idx += 1
            
        curr_mean = means[min(mean_idx, len(means)-1)]
        prev_mean = means[min(mean_idx-1, len(means)-1)] if mean_idx > 0 else mu_1
        
        # AR(1) process
        timeseries[t] = curr_mean + phi * (timeseries[t-1] - prev_mean) + np.random.normal(0, sigma_i)
        parameters[t] = curr_mean
    
    return timeseries, changepoints, parameters.reshape(-1, 1)


def generate_changepoint_variance_data(sigma_1=1, mu=0, nu=50, buffer_period=100, M=5000, 
                                     S=[0.25, 0.5, 1, 3], theta_values=[1, -1],
                                     starting_dist='normal', starting_params=None):
    """Generate time series with changepoints in variance.
    """
    
    # Set default parameters for starting distribution
    if starting_params is None:
        if starting_dist == 'normal':
            starting_params = {'loc': mu, 'scale': sigma_1}
    
    # Handle M=0 case: generate time series with no changepoints
    if M == 0:
        total_length = buffer_period + np.random.poisson(nu)
        timeseries = _generate_from_distribution(starting_dist, total_length, starting_params)
        parameters = np.full((total_length, 1), sigma_1)
        return timeseries, [], parameters
    
    # Generate changepoints: tau_i = tau_{i-1} + buffer_period + xi_i
    changepoints = []
    for i in range(M):
        xi_i = np.random.poisson(nu) # random interval
        if i == 0:
            tau_i = buffer_period + xi_i # buffer period + random interval
        else:
            tau_i = changepoints[i-1] + buffer_period + xi_i # changepoint + buffer period + random interval
        changepoints.append(tau_i)
    
    # Generate variances: sigma_i = sigma_{i-1} + theta * delta_i
    sigmas = [sigma_1]
    for i in range(1, M):
        delta_i = np.random.choice(S) # choose a random increment using uniform distribution over the set of valid increments
        theta = np.random.choice(theta_values) # theta allows us to choose the direction of the change
        sigma_i = max(0.1, sigmas[i-1] + theta * delta_i) # ensure sigma remains positive with minimum value
        sigmas.append(sigma_i)
    
    # Generate the timeseries data
    total_length = changepoints[-1] + buffer_period + np.random.poisson(nu)
    timeseries = np.zeros(total_length)
    parameters = np.zeros(total_length)
    
    # Generate data for each segment
    # Segment 0: from start to first changepoint (use starting distribution)
    if changepoints[0] > 0:
        timeseries[0:changepoints[0]] = _generate_from_distribution(starting_dist, changepoints[0], starting_params)
        parameters[0:changepoints[0]] = sigma_1
    
    # Segments 1 to M: from each changepoint to the next (use normal distribution with changing variances)
    for i in range(M):
        start_idx = changepoints[i]
        end_idx = changepoints[i + 1] if i + 1 < len(changepoints) else total_length
        
        if start_idx < end_idx:
            segment_length = end_idx - start_idx
            timeseries[start_idx:end_idx] = np.random.normal(mu, sigmas[i], segment_length)
            parameters[start_idx:end_idx] = sigmas[i]
    
    parameters = parameters.reshape(-1, 1)
    
    return timeseries, changepoints, parameters


def generate_changepoint_ar_variance_data(sigma_1=1, mu=0, nu=50, buffer_period=100, M=5000, 
                                        S=[0.25, 0.5, 1, 3], theta_values=[1, -1], phi=0.7,
                                        starting_dist='normal', starting_params=None):
    """Generate AR(1) time series with changepoints in variance.
    """
    # Set default parameters for starting distribution
    if starting_params is None:
        if starting_dist == 'normal':
            starting_params = {'loc': mu, 'scale': sigma_1}
    
    # Handle M=0 case: generate AR(1) time series with no changepoints
    if M == 0:
        total_length = buffer_period + np.random.poisson(nu)
        timeseries = np.zeros(total_length)
        parameters = np.full((total_length, 1), sigma_1)
        
        timeseries[0] = _generate_from_distribution(starting_dist, 1, starting_params)[0]
        for t in range(1, total_length):
            timeseries[t] = mu + phi * (timeseries[t-1] - mu) + np.random.normal(0, sigma_1)
        
        return timeseries, [], parameters
    
    # Generate changepoints
    changepoints = []
    for i in range(M):
        xi = np.random.poisson(nu)
        tau = buffer_period + xi if i == 0 else changepoints[-1] + buffer_period + xi
        changepoints.append(tau)
    
    # Generate variances
    sigmas = [sigma_1]
    for i in range(1, M):
        delta = np.random.choice(S)
        theta = np.random.choice(theta_values)
        sigma_i = max(0.1, sigmas[-1] + theta * delta)  # ensure sigma remains positive
        sigmas.append(sigma_i)
    
    # Generate time series
    total_length = changepoints[-1] + buffer_period + np.random.poisson(nu)
    timeseries = np.zeros(total_length)
    parameters = np.zeros(total_length)
    
    timeseries[0] = _generate_from_distribution(starting_dist, 1, starting_params)[0]
    parameters[0] = sigma_1
    
    sigma_idx = 0
    for t in range(1, total_length):
        if sigma_idx < len(changepoints) and t > changepoints[sigma_idx]:
            sigma_idx += 1
            
        curr_sigma = sigmas[min(sigma_idx, len(sigmas)-1)]
        
        # AR(1) process with changing variance
        timeseries[t] = mu + phi * (timeseries[t-1] - mu) + np.random.normal(0, curr_sigma)
        parameters[t] = curr_sigma
    
    return timeseries, changepoints, parameters.reshape(-1, 1)

