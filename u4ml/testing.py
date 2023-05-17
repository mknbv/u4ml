""" Defies test case for testing models written in pytorch. """
import random
from unittest import TestCase
import numpy as np
import numpy.testing as nt
import torch


def _np(tensor):
  """ Converts tensor to numpy array. """
  if isinstance(tensor, torch.Tensor):
    return tensor.cpu().detach().numpy()
  return tensor



class TorchTestCase(TestCase):
  """ Test case for testing code with models written in pytorch. """
  def setUp(self):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

  # pylint: disable=invalid-name
  def assertAllEqual(self, actual, expected):
    """ Checks that all elements of actual are equal to expected. """
    nt.assert_array_equal(_np(actual), _np(expected))

  def assertAllClose(self, actual, expected, rtol=1e-7, atol=0.):
    """ Checks that actual and expected arrays or torch tensors are equal. """
    nt.assert_allclose(_np(actual), _np(expected), rtol=rtol, atol=atol)

  # pylint: enable=invalid-name
