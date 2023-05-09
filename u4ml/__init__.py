""" u4ml package initialization. """

from u4ml.argparse import (
    get_defaults_parser,
    log_args,
)
from u4ml.plot import (
    extend_line,
    refresh_axis,
    set_figure_settings,
    plot_median_quantiles,
    semilogy_median_quantiles,
    semilogx_median_quantiles,
    loglog_median_quantiles,
    plot_mean_std,
    semilogy_mean_std,
    semilogx_mean_std,
    loglog_mean_std,
    mean_std_errorbar,
    LinesPlotter,
)

# pylint: disable=ungrouped-imports
HAVE_TORCH = True
try:
  import torch
except ModuleNotFoundError:
  HAVE_TORCH = False
if HAVE_TORCH:
  from u4ml.anneal import (
      AnnealingVariable,
      TorchSched,
      LinearAnneal,
  )
del HAVE_TORCH

HAVE_TF = True
try:
  import tensorflow as tf
except ModuleNotFoundError:
  HAVE_TF = False
if HAVE_TF:
  from u4ml.ptbe import (
      read_events,
      read_tag,
      plot_tag,
  )
  del tf
del HAVE_TF
# pylint: enable=ungrouped-imports
