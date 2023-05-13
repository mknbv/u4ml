""" Plot utilities. """
from contextlib import contextmanager, nullcontext
from collections import defaultdict
from functools import partial
from IPython.display import display, clear_output
from IPython import get_ipython
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mstats


def extend_line(line, newxs, newys):
  """ Updates line with new points. """
  xs, ys = map(list, line.get_data())
  xs.extend(newxs)
  ys.extend(newys)
  line.set_data(xs, ys)
  return line


def refresh_axis(ax=None):
  """ Refreshes axis. """
  if ax is None:
    ax = plt.gca()
  ax.relim()
  ax.autoscale_view()


def set_figure_settings(xlabel, ylabel, grid=True, title=None, legend=True):
  """ Sets up current figure. """
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid(grid)
  if title is not None:
    plt.title(title)
  if legend:
    plt.legend()


# pylint: disable=too-many-arguments
def _plot_median_quantiles(xplot_fn, ys, probs=(0.05, 0.95), axis=0,
                           label=None, alpha=0.1, **kwargs):
  """ Plots median and quantiles. """
  if len(probs) != 2:
    raise ValueError(f"probs must have length 2, got len(probs)={len(probs)}")
  probs = [probs[0], 0.5, probs[1]]
  quantiles = mstats.mquantiles(ys, probs, axis=axis)
  line, = xplot_fn(quantiles[(slice(None),) * axis + (1,)], label=label, **kwargs)
  xs, = xplot_fn.args
  fill = plt.fill_between(xs, quantiles[(slice(None),) * axis + (0,)],
                          quantiles[(slice(None),) * axis + (2,)],
                          color=line.get_color(), alpha=alpha)
  return line, fill


def plot_median_quantiles(x, ys, probs=(0.05, 0.95), axis=0,
                          label=None, alpha=0.1, **kwargs):
  """ Plot quantiles. """
  return _plot_median_quantiles(partial(plt.plot, x), ys, probs,
                                axis, label, alpha, **kwargs)

def semilogy_median_quantiles(x, ys, probs=(0.05, 0.95), axis=0,
                              label=None, alpha=0.1, **kwargs):
  """ Y-axis log plot of quantiles. """
  return _plot_median_quantiles(partial(plt.semilogy, x), ys, probs,
                                axis, label, alpha, **kwargs)

def semilogx_median_quantiles(x, ys, probs=(0.05, 0.95), axis=0,
                              label=None, alpha=0.1, **kwargs):
  """ X-axis log plot of quantiles. """
  _plot_median_quantiles(partial(plt.semilogx, x), ys, probs,
                         axis, label, alpha, **kwargs)

def loglog_median_quantiles(x, ys, probs=(0.05, 0.95), axis=0,
                            label=None, alpha=0.1, **kwargs):
  """ Log-log plot of quantiles. """
  _plot_median_quantiles(partial(plt.loglog, x), ys, probs,
                         axis, label, alpha, **kwargs)


def _plot_mean_std(xplot_fn, ys, axis=0, ylim=(-np.inf, np.inf),
                   label=None, alpha=0.1, **kwargs):
  """ Plots mean and standard deviation area. """
  mean = np.mean(ys, axis)
  std = np.std(ys, axis)
  line, = xplot_fn(mean, label=label, **kwargs)
  x, = xplot_fn.args
  fill_result = plt.fill_between(x,
                                 np.maximum(ylim[0], mean - std),
                                 np.minimum(ylim[1], mean + std),
                                 color=line.get_color(),
                                 alpha=alpha)
  return line, fill_result



def plot_mean_std(x, ys, axis=0, ylim=(-np.inf, np.inf),
                  label=None, alpha=0.1, **kwargs):
  """ Plots mean and standard deviation area. """
  return _plot_mean_std(partial(plt.plot, x), ys, axis, ylim=ylim,
                        label=label, alpha=alpha, **kwargs)

def semilogy_mean_std(x, ys, axis=0, ylim=(-np.inf, np.inf),
                      label=None, alpha=0.1, **kwargs):
  """ Y-axis log plot of mean and standard deviation area. """
  return _plot_mean_std(partial(plt.semilogy, x), ys, axis, ylim=ylim,
                        label=label, alpha=alpha, **kwargs)

def semilogx_mean_std(x, ys, axis=0, ylim=(-np.inf, np.inf),
                      label=None, alpha=0.1, **kwargs):
  """ X-axis log plot of mean and standard deviation area. """
  return _plot_mean_std(partial(plt.semilogx, x), ys, axis, ylim=ylim,
                        label=label, alpha=alpha, **kwargs)

def loglog_mean_std(x, ys, axis=0, ylim=(-np.inf, np.inf),
                    label=None, alpha=0.1, **kwargs):
  """ Log-log plot of mean and standard deviation area. """
  return _plot_mean_std(partial(plt.loglog, x), ys, axis, ylim=ylim,
                        label=label, alpha=alpha, **kwargs)


def _plot_mean_lines(plot_fn, ys, axis=0, label=None, alpha=0.1):
  """ Plots all lines and their mean using plot_fn. """
  ys = np.asarray(ys)
  lines = list(plot_fn(np.mean(ys, axis), label=label, lw=3))
  num_lines = ys.shape[axis]
  for i in range(num_lines):
    lines.append(
        plot_fn(np.take(ys, i, axis),
                color=lines[0].get_color(), alpha=alpha)[0]
    )
  return lines

def plot_mean_lines(x, ys, axis=0, label=None, alpha=0.1):
  """ Plots all lines and their mean. """
  return _plot_mean_lines(partial(plt.plot, x), ys, axis, label, alpha)

def semilogy_mean_lines(x, ys, axis=0, label=None, alpha=0.1):
  """ Y-axis log plot of lines and their mean. """
  return _plot_mean_lines(partial(plt.semilogy, x), ys, axis, label, alpha)

def semilogx_mean_lines(x, ys, axis=0, label=None, alpha=0.1):
  """ X-axis log plot of lines and their mean. """
  return _plot_mean_lines(partial(plt.semilogx, x), ys, axis, label, alpha)

def loglog_mean_lines(x, ys, axis=0, label=None, alpha=0.1):
  """ Log-log plot of lines and their mean. """
  return _plot_mean_lines(partial(plt.loglog, x), ys, axis, label, alpha)


def mean_std_errorbar(x, ys, axis=0, **kwargs):
  """ Plots mean and standard deviation error bars around it. """
  mean, std = np.mean(ys, axis), np.std(ys, axis)
  return plt.errorbar(x, mean, yerr=std, **kwargs)


# pylint: enable=too-many-arguments
def is_jupyter():
  """ Returns true if module is imported in Jupyter. """
  # pylint: disable=broad-except
  try:
    ipystr = str(type(get_ipython()))
  except Exception:
    return False
  # pylint: enable=broad-except
  return "zmqshell" in ipystr


class Output(widgets.Output):
  """ Same as ipywidget.Output but propagates exceptions. """
  def __exit__(self, exc_type, exc_val, exc_tb):
    super().__exit__(exc_type, exc_val, exc_tb)
    return False


class LinesPlotter:
  """ Iteratively plots several lines. """
  def __init__(self, ax, output=None):
    if output is None:
      output = Output() if is_jupyter() else None
    self.output = output
    self.ax = ax
    self.lines = {}

  @classmethod
  @contextmanager
  def make_autoclear_context(cls, ax=None, output=None):
    """ Creates instance and clears output on exiting the context. """
    instance = cls(ax if ax is not None else plt.gca(), output)
    with instance.autoclear_context():
      try:
        yield instance
      finally:
        pass

  @contextmanager
  def autoclear_context(self):
    """ Displays and clears the output on closing this context. """
    if self.output is not None:
      display(self.output)
    try:
      yield self
    finally:
      self.clear_output()

  def redraw_legend(self):
    """ Redraws of the current figure. """
    legend = self.ax.get_legend()
    if legend is not None:
      legend.remove()
    self.ax.legend()

  @contextmanager
  def context(self):
    """ Context of this plotter. """
    context = self.output if self.output else nullcontext()
    with context:
      try:
        yield
      finally:
        pass

  def plot_line(self, key, xs, ys, **kwargs):
    """ Creates a new line unde specified key. """
    with self.context():
      self.lines[key], = self.ax.plot(xs, ys, label=key, **kwargs)
      self.redraw_legend()
      return self.lines[key]

  def before_extend_line(self):
    """ Clears output (called before extending line). """
    with self.context():
      if self.output:
        clear_output(True)

  def extend_line(self, line, newxs, newys):
    """ Extends the given line. """
    with self.context():
      extend_line(line, newxs, newys)
      refresh_axis(ax=self.ax)

  def after_extend_line(self):
    """ Refreshes plot (called after extending line). """
    with self.context():
      if self.output:
        display(self.ax.get_figure())

  def extend(self, key, newxs, newys):
    """ Extends line with new values. """
    if key not in self.lines:
      self.plot_line(key, newxs, newys)
      return
    self.before_extend_line()
    self.extend_line(self.lines[key], newxs, newys)
    self.after_extend_line()

  def clear_output(self):
    """ Clears output -- call at the end of plotting to avoid duplicates. """
    with self.context():
      if self.output:
        clear_output()


class MeansPlotter:
  """ Plots lines, then means. """
  def __init__(self, lines_plotter):
    self.lines_plotter = lines_plotter
    self.lines = defaultdict(list)

  @classmethod
  @contextmanager
  def make_autoclear_context(cls, ax=None, output=None):
    """ Creates instance and clears output on exiting the context. """
    with LinesPlotter.make_autoclear_context(ax=ax, output=output) as plotter:
      instance = MeansPlotter(plotter)
      try:
        yield instance
      finally:
        pass

  def extend(self, key, newxs, newys):
    """ Extends a line with new values. """
    if key in self.lines and key not in self.lines_plotter.lines:
      self.lines_plotter.plot_line(key, [], [], linestyle="--",
                                   color=self.lines[key][0].get_color())
    self.lines_plotter.extend(key, newxs, newys)

  def clear_ax(self):
    """ Clears the underlying axis. """
    for child in self.lines_plotter.ax.get_children():
      try:
        child.remove()
      except NotImplementedError:
        pass
    if legend := self.lines_plotter.ax.get_legend():
      legend.remove()

  def means(self, clear_ax=True, **kwargs):
    """ Finishes all lines in the underlying plotter and plots means. """
    if clear_ax:
      self.clear_ax()
    for key, pltr_line in self.lines_plotter.lines.items():
      self.lines[key].append(pltr_line)
    plt.sca(self.lines_plotter.ax)
    for key, lines in self.lines.items():
      plot_mean_std(lines[0].get_data()[0],
                    [aline.get_data()[1] for aline in lines],
                    color=lines[0].get_color(), label=key, **kwargs)
    self.lines_plotter.lines.clear()
    self.lines_plotter.redraw_legend()
