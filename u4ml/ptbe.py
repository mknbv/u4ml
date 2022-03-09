""" Utilities for plotting tensorboard events. """
from collections import defaultdict, OrderedDict
from operator import itemgetter
import os
import matplotlib.pyplot as plt
import tensorflow as tf


class CacheDict(OrderedDict):
  """ Cache dictionary of maximum size. """
  def __init__(self, maxsize, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.maxsize = maxsize

  def __setitem__(self, key, value):
    if len(self) == self.maxsize:
      self.popitem(last=False)
    super().__setitem__(key, value)


_CACHE = CacheDict(100)


def cache_pop(key):
  """ Pops element under key from the cache. """
  return _CACHE.pop(key)

def cache_get(key, default=None):
  """ Returns value under key from cache or default in case of lookup error. """
  return _CACHE.get(key, default)



def read_events(event_filename, tag=None, data=None, purge_orphaned=True):
  """ Reads all events from the event file. """
  data = data if data is not None else defaultdict(dict)
  for event in tf.compat.v1.train.summary_iterator(event_filename):
    # pylint: disable=no-member
    if (purge_orphaned
        and event.session_log.status == tf.compat.v1.SessionLog.START):
      # pylint: enable=no-member
      for key in data.keys():
        data[key] = {
            step_time: val
            for step_time, val in data[tag].items()
            if step_time[0] < event.step
        }

    value_iterator = (event.summary.value if tag is None
                      else filter(lambda v: v.tag == tag, event.summary.value))
    for val in value_iterator:
      data[val.tag][(event.step, event.wall_time)] = val.simple_value
  return data


def read_tag(path, tag, use_cache=True):
  """ Reads and returns steps and values for a specific tag. """
  if use_cache and (path, tag) in _CACHE:
    return _CACHE[(path, tag)]
  events = defaultdict(dict)
  for fname in filter(lambda fname: fname.startswith("events"),
                      os.listdir(path)):
    events = read_events(os.path.join(path, fname), tag=tag, data=events)
  steps, values = (list(map(itemgetter(0), events[tag].keys())),
                   list(events[tag].values()))
  if use_cache:
    _CACHE[(path, tag)] = steps, values
  return steps, values


def plot_tag(path, tag, use_cache=True, ax=None, **kwargs):
  """ Plots the steps and values under the given tag. """
  steps, vals = read_tag(path, tag, use_cache=use_cache)
  if ax is None:
    ax = plt.gca()
  ax.plot(steps, vals, **kwargs)
