def GMM(observations):
    # Create a mixture of two Gaussians:
    import tensorflow as tf
    import numpy as np
    import tensorflow_probability as tfp
    from tensorflow_probability import edward2 as ed
    import functools
    import scipy.stats as stat

    tfd = tfp.distributions
    tfb = tfp.bijectors
    dtype = np.float32
    dims = 2
    components = 3

    def session_options(enable_gpu_ram_resizing=True):
        """Convenience function which sets common `tf.Session` options."""
        config = tf.ConfigProto()
        config.log_device_placement = True
        if enable_gpu_ram_resizing:
            # `allow_growth=True` makes it possible to connect multiple colabs to your
            # GPU. Otherwise the colab malloc's all GPU ram.
            config.gpu_options.allow_growth = True
        return config

    def reset_sess(config=None):
        """Convenience function to create the TF graph and session, or reset them."""
        if config is None:
            config = session_options()
        tf.reset_default_graph()
        global sess
        try:
            sess.close()
        except:
            pass
        sess = tf.InteractiveSession(config=config)

    reset_sess()

    class MVNCholPrecisionTriL(tfd.TransformedDistribution):
        """MVN from loc and (Cholesky) precision matrix."""

        def __init__(self, loc, chol_precision_tril, name=None):
            super(MVNCholPrecisionTriL, self).__init__(
                distribution=tfd.Independent(tfd.Normal(tf.zeros_like(loc),
                                                        scale=tf.ones_like(loc)),
                                             reinterpreted_batch_ndims=1),
                bijector=tfb.Chain([
                    tfb.Affine(shift=loc),
                    tfb.Invert(tfb.Affine(scale_tril=chol_precision_tril,
                                          adjoint=True)),
                ]),
                name=name)

    def joint_log_prob(observations, mix_probs, loc, chol_precision):
        """BGMM with priors: loc=Normal, precision=Inverse-Gamma, mix=Dirichlet.

        Args:
          observations: `[n, d]`-shaped `Tensor` representing Bayesian Gaussian
            Mixture model draws. Each sample is a length-`d` vector.
          mix_probs: `[K]`-shaped `Tensor` representing random draw from
            `SoftmaxInverse(Dirichlet)` prior.
          loc: `[K, d]`-shaped `Tensor` representing the location parameter of the
            `K` components.
          chol_precision: `[K, d, d]`-shaped `Tensor` representing `K` lower
            triangular `cholesky(Precision)` matrices, each being sampled from
            a Wishart distribution.

        Returns:
          log_prob: `Tensor` representing joint log-density over all inputs.
        """
        rv_observations = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mix_probs),
            components_distribution=MVNCholPrecisionTriL(
                loc=loc,
                chol_precision_tril=chol_precision))
        log_prob_parts = [
            rv_observations.log_prob(observations),  # Sum over samples.
            pi.log_prob(mix_probs)[..., tf.newaxis],
            mu.log_prob(loc),  # Sum over components.
            sigma.log_prob(chol_precision),  # Sum over components.
        ]
        sum_log_prob = tf.reduce_sum(tf.concat(log_prob_parts, axis=-1), axis=-1)
        # Note: for easy debugging, uncomment the following:
        #sum_log_prob = tf.Print(sum_log_prob, log_prob_parts)
        return sum_log_prob

    pi = tfd.Dirichlet(
        concentration=np.ones(components, dtype) * 100.,
        name='pi_dist')

    mu = tfd.Independent(
        tfd.Normal(
            loc=np.stack([np.zeros(dims, dtype)]*components),
            scale=tf.ones([components, dims], dtype)),
        reinterpreted_batch_ndims=1,
        name='mu_dist')

    sigma = tfd.Wishart(
        df=5,
        scale_tril=np.stack([np.eye(dims, dtype=dtype)] * components),
        input_output_cholesky=True,
        name='sigma_dist')

    # testing
    num_samples = 1000
    true_loc = np.array([[-2, -2],
                         [0, 0],
                         [2, 2]], dtype)
    random = np.random.RandomState(seed=42)

    true_hidden_component = random.randint(0, components, num_samples)
    observations = (true_loc[true_hidden_component] +
                    random.randn(num_samples, dims).astype(dtype))
    unnormalized_posterior_log_prob = functools.partial(joint_log_prob, observations)

    initial_state = [
        tf.fill([components],
                value=np.array(1. / components, dtype),
                name='mix_probs'),
        tf.constant(np.array([[-2, -2],
                              [0, 0],
                              [-20, 1]], dtype),
                    name='loc'),
        tf.eye(dims, batch_shape=[components], dtype=dtype, name='chol_precision'),
    ]
    unconstraining_bijectors = [
        tfb.SoftmaxCentered(),
        tfb.Identity(),
        tfb.Chain([
            tfb.TransformDiagonal(tfb.Softplus()),
            tfb.FillTriangular(),
        ])]
    step_size = tf.get_variable(
        name='step_size',
        initializer=np.array(0.05, dtype),
        use_resource=True,
        trainable=False)
    [mix_probs, loc, chol_precision], kernel_results = tfp.mcmc.sample_chain(
        num_results=5000,
        num_burnin_steps=500,
        current_state=initial_state,
        kernel=tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_posterior_log_prob,
                step_size=step_size,
                step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(),
                num_leapfrog_steps=5),
            bijector=unconstraining_bijectors))

    acceptance_rate = tf.reduce_mean(tf.to_float(kernel_results.inner_results.is_accepted))
    mean_mix_probs = tf.reduce_mean(mix_probs, axis=0)
    mean_loc = tf.reduce_mean(loc, axis=0)
    mean_chol_precision = tf.reduce_mean(chol_precision, axis=0)


    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    [
        acceptance_rate_,
        mean_mix_probs_,
        mean_loc_,
        mean_chol_precision_,
        mix_probs_,
        loc_,
        chol_precision_,
    ] = sess.run([
        acceptance_rate,
        mean_mix_probs,
        mean_loc,
        mean_chol_precision,
        mix_probs,
        loc,
        chol_precision,
    ])
    print(mean_mix_probs_)
    print(mean_loc_)
    print(acceptance_rate_)
    word_label = []
    for obs in observations:
        max = -1;
        maxi = -1
        for comp in range(components):
            prob = mean_mix_probs_[comp] * stat.multivariate_normal(mean_loc_[comp], mean_chol_precision_[comp]).pdf(
                obs)
            if (prob > max):
                max = prob;
                maxi = comp
        word_label.append(maxi)
    for i in range(components):
        print(i, word_label.count(i))

if __name__ == '__main__':
    # Create a mixture of two Gaussians:
    import tensorflow as tf
    import numpy as np
    import tensorflow_probability as tfp
    from tensorflow_probability import edward2 as ed
    import functools
    import pickle
    import time
    import scipy.stats as stat

    #GMM()
    #quit()
    fname = "./variational0.dat"
    T = 31
    K = 100
    with open(fname,"rb") as f:
        fit = pickle.load(f)
    rho_t = {}

    for t in range(T):
        rho_t[t] = fit['rho_' + str(t)]
    alpha = fit['alpha']


    tfd = tfp.distributions
    tfb = tfp.bijectors
    dtype = np.float32
    dims = rho_t[0].shape[1]
    components = 3

    def session_options(enable_gpu_ram_resizing=True):
        """Convenience function which sets common `tf.Session` options."""
        config = tf.ConfigProto(allow_soft_placement=True)
        config.log_device_placement = True


        if enable_gpu_ram_resizing:
            # `allow_growth=True` makes it possible to connect multiple colabs to your
            # GPU. Otherwise the colab malloc's all GPU ram.
            config.gpu_options.allow_growth = True
        return config


    def reset_sess(config=None):
        """Convenience function to create the TF graph and session, or reset them."""
        if config is None:
            config = session_options()
        tf.reset_default_graph()
        global sess
        try:
            sess.close()
        except:
            pass
        sess = tf.InteractiveSession(config=config)


    reset_sess()


    class MVNCholPrecisionTriL(tfd.TransformedDistribution):
        """MVN from loc and (Cholesky) precision matrix."""

        def __init__(self, loc, chol_precision_tril, name=None):
            super(MVNCholPrecisionTriL, self).__init__(
                distribution=tfd.Independent(tfd.Normal(tf.zeros_like(loc),
                                                        scale=tf.ones_like(loc)),
                                             reinterpreted_batch_ndims=1),
                bijector=tfb.Chain([
                    tfb.Affine(shift=loc),
                    tfb.Invert(tfb.Affine(scale_tril=chol_precision_tril,
                                          adjoint=True)),
                ]),
                name=name)

    def joint_log_prob(observations, mix_probs, loc, chol_precision):
        """BGMM with priors: loc=Normal, precision=Inverse-Gamma, mix=Dirichlet.

        Args:
          observations: `[n, d]`-shaped `Tensor` representing Bayesian Gaussian
            Mixture model draws. Each sample is a length-`d` vector.
          mix_probs: `[K]`-shaped `Tensor` representing random draw from
            `SoftmaxInverse(Dirichlet)` prior.
          loc: `[K, d]`-shaped `Tensor` representing the location parameter of the
            `K` components.
          chol_precision: `[K, d, d]`-shaped `Tensor` representing `K` lower
            triangular `cholesky(Precision)` matrices, each being sampled from
            a Wishart distribution.

        Returns:
          log_prob: `Tensor` representing joint log-density over all inputs.
        """
        rv_observations = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mix_probs),
            components_distribution=MVNCholPrecisionTriL(
                loc=loc,
                chol_precision_tril=chol_precision))
        log_prob_parts = [
            rv_observations.log_prob(observations),  # Sum over samples.
            pi.log_prob(mix_probs)[..., tf.newaxis],
            mu.log_prob(loc),  # Sum over components.
            sigma.log_prob(chol_precision),  # Sum over components.
        ]
        sum_log_prob = tf.reduce_sum(tf.concat(log_prob_parts, axis=-1), axis=-1)
        # Note: for easy debugging, uncomment the following:
        #sum_log_prob = tf.Print(sum_log_prob, log_prob_parts)
        return sum_log_prob

    pi = tfd.Dirichlet(
        concentration=np.ones(components, dtype) * 100.,
        name='pi_dist')

    mu = tfd.Independent(
        tfd.Normal(
            loc=np.stack([np.zeros(dims, dtype)]*components),
            scale=tf.ones([components, dims], dtype)),
        reinterpreted_batch_ndims=1,
        name='mu_dist')

    sigma = tfd.Wishart(
        df=dims,
        scale_tril=np.stack([np.eye(dims, dtype=dtype)] * components),
        input_output_cholesky=True,
        name='sigma_dist')

    #testing
    """
    num_samples = 1000
    true_loc = np.array([[-2, -2],
                         [0, 0],
                         [2, 2]], dtype)
    random = np.random.RandomState(seed=42)

    true_hidden_component = random.randint(0, components, num_samples)
    observations = (true_loc[true_hidden_component] +
                    random.randn(num_samples, dims).astype(dtype))
    print(observations.shape)
    """
    for t_idx in range(T):
        stamp = time.time()
        observations = rho_t[t_idx]
        unnormalized_posterior_log_prob = functools.partial(joint_log_prob, observations)
        initial_state = [
            tf.fill([components],
                    value=np.array(1. / components, dtype),
                    name='pi'),
            tf.constant(np.stack([np.ones(dims, dtype)]*components),
                        name='mu'),
            tf.eye(dims, batch_shape=[components], dtype=dtype, name='sigma'),
        ]
        unconstraining_bijectors = [
            tfb.SoftmaxCentered(),
            tfb.Identity(),
            tfb.Chain([
                tfb.TransformDiagonal(tfb.Softplus()),
                tfb.FillTriangular(),
            ])]

        step_size = tf.get_variable(
            name='step_size',
            initializer=np.array(0.05, dtype),
            use_resource=True,
            trainable=False)
        [mix_probs, loc, chol_precision], kernel_results = tfp.mcmc.sample_chain(
            num_results=3000,
            num_burnin_steps=500,
            current_state=initial_state,
            kernel=tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=unnormalized_posterior_log_prob,
                    step_size=step_size,
                    step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(),
                    num_leapfrog_steps=5),
                bijector=unconstraining_bijectors))

        acceptance_rate = tf.reduce_mean(tf.to_float(kernel_results.inner_results.is_accepted))
        mean_mix_probs = tf.reduce_mean(mix_probs, axis=0)
        mean_loc = tf.reduce_mean(loc, axis=0)
        mean_chol_precision = tf.reduce_mean(chol_precision, axis=0)

        obs = tf.convert_to_tensor(observations)
        theta = tfd.MultivariateNormalTriL(loc=mean_loc, scale_tril=mean_chol_precision)
        c = theta.prob(observations[0])
        #c = tf.get_variable("c", [len(observations)], dtype=dtype, initializer=tf.zeros_initializer)
        tf.get_variable_scope().reuse_variables()
        condition = lambda i, px: i < len(observations)

        '''
        

        def c_assign(i, c):
            theta = tfd.MultivariateNormalTriL(loc=mean_loc, scale_tril=mean_chol_precision)
            c = tf.get_variable("c")
            c = tf.scatter_update(
                c,
                indices=i,
                updates=tf.to_float(tf.argmax(theta.prob(obs[i])))
            )

            return i + 1, c


        i, c = tf.while_loop(condition, c_assign, [0, c])
        '''
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        [
            acceptance_rate_,
            mean_mix_probs_,
            mean_loc_,
            mean_chol_precision_,
            mix_probs_,
            loc_,
            chol_precision_,
            c_
        ] = sess.run([
            acceptance_rate,
            mean_mix_probs,
            mean_loc,
            mean_chol_precision,
            mix_probs,
            loc,
            chol_precision,
            c
        ])
        print(acceptance_rate_)
        print(mean_loc_)
        print(mean_mix_probs_)
        print(c_)
        c_ = c_.tolist()
        for i in range(components):
            print(i, c_.count(i))
        break