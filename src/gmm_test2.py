import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
tfd = tfp.distributions
tfb = tfp.bijectors
dtype = np.float32

def gmm_model(dimension, components):
    pi = ed.Dirichlet(np.ones(components, dtype), sample_shape=[components], name="pi")
    mu = ed.Normal(
            loc=np.stack([np.zeros(dimension, dtype)]*components),
            scale=tf.ones([components, dimension], dtype),name="mu")
    sigma = ed.InverseGamma(concentration=tf.ones(dimension), rate=tf.ones(dimension), name='sigma')
    gm = ed.MixtureSameFamily(
        mixture_distribution=pi,
        components_distribution=tfd.Normal(
            loc=mu,  # One for each component.
            scale=sigma))  # And same here.
    return gm

gmm = gmm_model(10,3)

quit()
dims = rho_t[0].shape[1]
components = 10

num_samples = 1000
true_loc = np.array([[-2, -2],
                     [0, 0],
                     [2, 2]], dtype)
random = np.random.RandomState(seed=42)

true_hidden_component = random.randint(0, components, num_samples)
observations = (true_loc[true_hidden_component] +
                random.randn(num_samples, dims).astype(dtype))
print(observations.shape)