
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import _Merge
from functools import partial

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples.
Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
between each pair of input points."""
    def __init__(self, batch_size, **kwargs):
        super(_Merge, self).__init__(**kwargs)
        self.batch_size = batch_size

    def _merge_function(self, inputs):
	alpha = K.random_uniform((self.batch_size, 1, 1, 1))
	return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def wasserstein_loss(y_true, y_pred):
    '''
    In a standard GAN, the discriminator has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function. Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0.
    '''

    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
Calculates the gradient penalty loss for a batch of "averaged" samples.
In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
this function at all points in the input space. The compromise used in the paper is to choose random points
on the lines between real and generated samples, and check the gradients at these points. Note that it is the
gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
Then we get the gradients of the discriminator w.r.t. the input averaged samples.
The l2 norm and penalty can then be calculated for this gradient.
Note that this loss function requires the original averaged samples as input, but Keras only supports passing
y_true and y_pred to loss functions. To get around this,  make a partial() of the function with the
averaged_samples argument, and use that for model training.
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
			      axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def the_Model(naked_model, input_shape):
    """Make a model from Hunters code work for Jordan's"""
    noise = Input(shape=input_shape)
    img = naked_model(noise)

    return Model(noise, img)


def get_generator(naked_generator, naked_critic, latent_dim, img_shape, optimizer):
    # make sure they're right
    naked_critic.trainable = False
    naked_generator.trainable = True
    # get models
    generator = the_Model(naked_generator, latent_dim)
    critic = the_Model(naked_critic, img_shape)

    # Sampled noise for input to generator
    z_gen = Input(shape=latent_dim)
    # Generate images based of noise
    img = generator(z_gen)
    # Discriminator determines validity
    valid = critic(img)
    # Defines generator model
    generator_model = Model(z_gen, valid)
    generator_model.compile(loss=wasserstein_loss, optimizer=optimizer,metrics=['accuracy'])

    return generator, generator_model


def get_critic(naked_generator, naked_critic, latent_dim, img_shape, batch_size, optimizer):
    # make sure they're right
    naked_critic.trainable = True
    naked_generator.trainable = False
    # get models
    generator = the_Model(naked_generator, latent_dim)
    critic = the_Model(naked_critic, img_shape)

    #Image input (real sample)
    real_img = Input(shape=img_shape)

    # Noise input
    z_disc = Input(shape=latent_dim)
    # Generate image based of noise (fake sample)
    fake_img = generator(z_disc)

    # Discriminator determines validity of the real and fake images
    fake = critic(fake_img)
    valid = critic(real_img)

    # Construct weighted average between real and fake images
    interpolated_img = RandomWeightedAverage(batch_size)([real_img, fake_img])
    # Determine validity of weighted sample
    validity_interpolated = critic(interpolated_img)

    # Use Python partial to provide loss function with additional
    # 'averaged_samples' argument. Kears only lets you have two arguments in loss functions.
    partial_gp_loss = partial(gradient_penalty_loss,
                      averaged_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

    critic_model = Model(inputs=[real_img, z_disc],
                        outputs=[valid, fake, validity_interpolated])
    critic_model.compile(loss=[wasserstein_loss,
                                    wasserstein_loss,
                                    partial_gp_loss],
                              optimizer=optimizer,
                              loss_weights=[1, 1, 10],metrics=['accuracy'])

    return critic, critic_model


