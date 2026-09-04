import numpy as np
import os
import sys
from libcuflynx.param_id.operation_funcs import series_to_constant
try:
    import sympy
except ImportError:
    # Not a declared dependency of libcuflynx (#435): nothing under src/libcuflynx imports it,
    # and the only users are the four RICRI terminal-frequency funcs below, whose closed-form
    # roots were *derived* with sympy rather than needing it at runtime. Anyone calling those
    # installs sympy themselves; everything else in this file works without it.
    sympy = None
from scipy.signal import find_peaks

from libcuflynx.param_id.differentiable import differentiable
from libcuflynx.param_id.math_backend import make_math_backend, bind_backend

mb = make_math_backend("numpy")
"""
"series_to_constant" decorator for functions that turn a series into a constant
Needed if you want to plot the series ontop of estimated constants

"differentiable" decorator for functions that are differentiable

"mb" is the math backend for optionally differentiable operations, can be "numpy" or "casadi"

This module is *library* code (issue #433): do not add your own operations here, because an
upgrade of libcuflynx replaces the file. Put them in your own file and name it with the
``operation_funcs_external_path`` config key -- they are merged into the same registry as the
operations below, decorators and all. See ``funcs_user/operation_funcs_example.py``.

Observable operations: every top-level function defined in this module is registered automatically,
except private names (leading ``_``), ``series_to_constant``, ``register_user_operations``, ``ml_to_m3``,
and names starting with ``RICRI_`` (CellML parameter helpers). Put other helpers in another module or
prefix with ``_``.
"""


def series_to_constant(func):
    func.series_to_constant = True
    return func

# example function
def ml_to_m3(x):
    return x*1e-6

def RICRI_get_pole_freq_1_Hz(C_T, I_1, I_2, R_T, frac_R_T_1_of_R_T):
    assert isinstance(C_T, float)
    assert isinstance(I_1, float)
    assert isinstance(I_2, float)
    assert isinstance(R_T, float)
    assert isinstance(frac_R_T_1_of_R_T, float)
    R_1 = R_T*frac_R_T_1_of_R_T
    R_2 = R_T*(1.0-frac_R_T_1_of_R_T)
    # this frequency was calculated using sympy from the differential equation for an RICRI terminal
    freq = np.float(0.159154943091895*sympy.re(sympy.Abs(R_2/(2*I_2) + sympy.sqrt(C_T*(C_T*R_2**2 - 4*I_2))/(2*C_T*I_2)).evalf()))
    assert isinstance(freq, float)
    return freq

def RICRI_get_pole_freq_2_Hz(C_T, I_1, I_2, R_T, frac_R_T_1_of_R_T):
    assert isinstance(C_T, float)
    assert isinstance(I_1, float)
    assert isinstance(I_2, float)
    assert isinstance(R_T, float)
    assert isinstance(frac_R_T_1_of_R_T, float)
    R_1 = R_T*frac_R_T_1_of_R_T
    R_2 = R_T*(1.0-frac_R_T_1_of_R_T)
    # this frequency was calculated using sympy from the differential equation for an RICRI terminal
    freq = np.float(0.159154943091895*sympy.re(sympy.Abs(R_2/(2*I_2) - sympy.sqrt(C_T*(C_T*R_2**2 - 4*I_2))/(2*C_T*I_2)).evalf()))
    assert isinstance(freq, float)
    return freq

def RICRI_get_zero_freq_1_Hz(C_T, I_1, I_2, R_T, frac_R_T_1_of_R_T):
    assert isinstance(C_T, float)
    assert isinstance(I_1, float)
    assert isinstance(I_2, float)
    assert isinstance(R_T, float)
    assert isinstance(frac_R_T_1_of_R_T, float)
    R_1 = R_T*frac_R_T_1_of_R_T
    R_2 = R_T*(1.0-frac_R_T_1_of_R_T)
    # this frequency was calculated using sympy from the differential equation for an RICRI terminal
    freq = np.float(0.159154943091895*sympy.re(sympy.Abs(((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))/(3*(sympy.sqrt(-4*((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))**3 + (2*(I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1**2*I_2**2))**2)/2 + (I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(2*C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(2*C_T*I_1**2*I_2**2))**(1/3)) + (sympy.sqrt(-4*((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))**3 + (2*(I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1**2*I_2**2))**2)/2 + (I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(2*C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(2*C_T*I_1**2*I_2**2))**(1/3)/3 + (I_1*R_2 + I_2*R_1)/(3*I_1*I_2)).evalf()))
    assert isinstance(freq, float)
    return freq

def RICRI_get_zero_freq_2_Hz(C_T, I_1, I_2, R_T, frac_R_T_1_of_R_T):
    assert isinstance(C_T, float)
    assert isinstance(I_1, float)
    assert isinstance(I_2, float)
    assert isinstance(R_T, float)
    assert isinstance(frac_R_T_1_of_R_T, float)
    R_1 = R_T*frac_R_T_1_of_R_T
    R_2 = R_T*(1.0-frac_R_T_1_of_R_T)
    # this frequency was calculated using sympy from the differential equation for an RICRI terminal
    freq = np.float(0.159154943091895*sympy.re(sympy.Abs(-((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))/(3*(-1/2 + sympy.sqrt(3)*sympy.I/2)*(sympy.sqrt(-4*((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))**3 + (2*(I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1**2*I_2**2))**2)/2 + (I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(2*C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(2*C_T*I_1**2*I_2**2))**(1/3)) - (-1/2 + sympy.sqrt(3)*sympy.I/2)*(sympy.sqrt(-4*((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))**3 + (2*(I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1**2*I_2**2))**2)/2 + (I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(2*C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(2*C_T*I_1**2*I_2**2))**(1/3)/3 - (I_1*R_2 + I_2*R_1)/(3*I_1*I_2)).evalf()))
    assert isinstance(freq, float)
    return freq

def RICRI_get_zero_freq_3_Hz(C_T, I_1, I_2, R_T, frac_R_T_1_of_R_T):
    assert isinstance(C_T, float)
    assert isinstance(I_1, float)
    assert isinstance(I_2, float)
    assert isinstance(R_T, float)
    assert isinstance(frac_R_T_1_of_R_T, float)
    R_1 = R_T*frac_R_T_1_of_R_T
    R_2 = R_T*(1.0-frac_R_T_1_of_R_T)
    # this frequency was calculated using sympy from the differential equation for an RICRI terminal
    freq = np.float(0.159154943091895*sympy.re(sympy.Abs(-((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))/(3*(-1/2 - sympy.sqrt(3)*sympy.I/2)*(sympy.sqrt(-4*((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))**3 + (2*(I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1**2*I_2**2))**2)/2 + (I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(2*C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(2*C_T*I_1**2*I_2**2))**(1/3)) - (-1/2 - sympy.sqrt(3)*sympy.I/2)*(sympy.sqrt(-4*((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))**3 + (2*(I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1**2*I_2**2))**2)/2 + (I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(2*C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(2*C_T*I_1**2*I_2**2))**(1/3)/3 - (I_1*R_2 + I_2*R_1)/(3*I_1*I_2)).evalf()))
    assert isinstance(freq, float)
    return freq

# TODO we should find a way to only find_peaks once per subexperiment
# ATM if multiple of the below functions are called, it does find_peaks multiple times
@series_to_constant
def calc_spike_period(t, V, series_output=False):
    if series_output:
        return V
    peak_idxs, peak_properties = find_peaks(V)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 2:
        # there aren't enough peaks to calculate a period
        # so set the period to the max time of the simulation
        period = t[-1] - t[0]
    else:
        # calculate the average period between peaks
        period = np.sum([t[peak_idxs[II+1]] - t[peak_idxs[II]] for II in range(len(peak_idxs)-1)])/(len(peak_idxs) - 1)
    return period

@series_to_constant
def calc_spike_frequency_windowed(t, V, series_output=False, spike_min_thresh=-10, start_frac=0.0, end_frac=1.0):
    """
    this calculates the number of spikes per 
    second in the given window. Not an accurate actual 
    frequency, but useful for some applications.

    This includes a minimum threshold for peaks of spike_min_thresh
    """
    if series_output:
        return V
    # get the start and end of the window
    start_idx = int(start_frac*(len(t)-1))
    end_idx = int(end_frac*(len(t)-1))
    peak_idxs, peak_properties = find_peaks(V[start_idx:end_idx], height=spike_min_thresh)

    # TODO maybe check peak properties here
    spikes_per_s = len(peak_idxs)/(t[end_idx] - t[start_idx])
    return spikes_per_s

@series_to_constant
def calc_spike_count_windowed(t, V, series_output=False, spike_min_thresh=-10,
                              start_frac=0.0, end_frac=1.0):
    """The number of spikes in the window -- the count itself, not a rate.

    Same peak detection as :func:`calc_spike_frequency_windowed`; that one divides by
    the window length and this one does not. The distinction matters to the cost: a
    count is what a Poisson likelihood scores (``poisson_MLE`` takes the observed count
    as ``prob_dist_params['k']`` and the model's count as lambda), and dividing by the
    window turns an integer into a rate whose Poisson variance is wrong by the window
    length squared.

    It also matters to an emulator. An integer-valued observable can be learned as a
    classifier over the values it takes rather than regressed as a continuous quantity
    -- see ``multi_phase_<name>`` -- and that is only visible if the observable is
    still an integer by the time the emulator sees it.
    """
    if series_output:
        return V
    start_idx = int(start_frac*(len(t)-1))
    end_idx = int(end_frac*(len(t)-1))
    peak_idxs, peak_properties = find_peaks(V[start_idx:end_idx], height=spike_min_thresh)
    return float(len(peak_idxs))


@series_to_constant
def first_peak_time(t, V, series_output=False, spike_min_thresh=None):
    """ 
    returns the time value (time from start of pre_time, NOT the start of 
    experiment or subexperiment) that the first peak occurs

    It is the time from the start, but it only checks in the subexperiment defined in obs_data.
    """
    if series_output:
        return V
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh)
    
    if len(peak_idxs) == 0:
        # there are no peaks, return the time of the subexperiment
        return t[-1]
    
    t_first_peak = t[peak_idxs[0]] # this is from the start of the pre_time, not the start of experiment.
    return t_first_peak

@series_to_constant
def first_peak_time_from_subexp_start(t, V, series_output=False, spike_min_thresh=None):
    """ 
    returns the time value (time from start of the subexp, NOT the start of 
    experiment) that the first peak occurs
    """
    if series_output:
        return V
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh)
    
    if len(peak_idxs) == 0:
        # there are no peaks, return the time of the subexperiment
        return t[-1]
    
    t_first_peak = t[peak_idxs[0]] - t[0] # this calcs from start of subexperiment but there are plotting issues
    return t_first_peak

@differentiable
@series_to_constant
def steady_state_min(x, series_output=False):
    """
    finds the min of the second half of this subexperiment. 
    The aim of this is to allow the dynamics to reach steady state
    or periodic steady state before getting the minimum
    """
    if series_output:
        return x
    else:
        return mb.min(x[len(x)//2:])
    
@differentiable
@series_to_constant
def steady_state_avg(x, series_output=False):
    """
    finds the average of the second half of this subexperiment. 
    The aim of this is to allow the dynamics to reach steady state
    or periodic steady state before getting the average
    """
    if series_output:
        return x
    else:
        return mb.mean(x[len(x)//2:])

@series_to_constant
def calc_min_to_max_period_diff(t, V, series_output=False, spike_min_thresh=None):
    if series_output:
        return V
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 2:
        # there aren't enough peaks to calculate a period
        # so set the period_diff to the max time of the simulation
        period_diff = t[-1] - t[0]
    else:
        # calculate the periods
        periods = [t[peak_idxs[II+1]] - t[peak_idxs[II]] for II in range(len(peak_idxs)-1)]
        # calculate the difference in time between max period and min period
        period_diff = max(periods) - min(periods)

    return period_diff

@series_to_constant
def calc_min_peak(t, V, series_output=False, spike_min_thresh=None):
    if series_output:
        return V
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 1:
        # if there aren't spikes set the min peak to the max of the voltage the max/sudo-peak)
        min_peak = max(V)
    else:
        min_peak = min(V[peak_idxs])

    return min_peak

@series_to_constant
def min_between_first_two_spikes(t, V, series_output=False, spike_min_thresh=None):
    """The afterhyperpolarisation trough: the minimum between a trace's first two spikes.

    Locates the spikes in *this* trace rather than taking a fixed window, so the recording
    and the model are each measured over their own first interspike interval. A window fixed
    from the recording would make the feature partly a spike-timing measurement: on one
    SN_full experiment the model had a spike inside the recording's window in only 10 of 40
    posterior draws, because it fires more slowly than the cell, and the "trough" was then a
    rising subthreshold ramp. Timing belongs in ``first_peak_time``; this is depth alone.

    With fewer than two spikes there is no interspike interval, and the maximum of the trace
    is returned -- the same sentinel :func:`calc_min_peak` uses. It is deliberately far from
    any real trough: paired with ``gaussian_MLE_robust`` the cost is then capped, rather than
    the subthreshold minimum being compared against a real AHP, which would duplicate what a
    steady-state minimum already measures.
    """
    if series_output:
        return V
    peak_idxs, _ = find_peaks(V, height=spike_min_thresh)
    if len(peak_idxs) < 2:
        return mb.max(V)
    return mb.min(V[peak_idxs[0]:peak_idxs[1] + 1])


@series_to_constant
def AHP_minus_steady_state_min(t, V, series_output=False, spike_min_thresh=None):
    """How much deeper the *first* afterhyperpolarisation is than the steady-state minimum.

    ``min_between_first_two_spikes(V) - steady_state_min(V)``, i.e. the trough after the first
    spike measured against the trough the trace settles to. Two reasons to score the
    difference rather than the trough itself:

    * **It is not redundant.** ``steady_state_min`` is the minimum of the second half, which
      during repetitive firing *is* a later AHP trough. On the SN_full recordings the first
      trough and the steady-state minimum sit 0.5-2.8 mV apart, well inside a 4 mV sigma, so
      an absolute AHP observable mostly repeats what ``steady_state_min`` already says.
      The difference is what is left over: spike-frequency accommodation of the trough.
    * **It barely charges the firing decision again.** A trace that never fires has its
      firing already scored by the spike counts and by any jump observable; a difference that
      is near zero whenever the model is silent does not charge it a third time. See #498's
      sibling discussion -- the same binary is otherwise asserted by ~22 observables that are
      all deterministic functions of it.

    A trace with fewer than two spikes has no first interspike interval and so no measurable
    accommodation: **zero** is returned, not the trace maximum that
    :func:`min_between_first_two_spikes` uses. That difference is deliberate. An absolute
    trough needs a sentinel far from any real value so a silent model is obviously wrong; a
    *difference* wants the opposite, because "the model did not fire" is already asserted by
    the spike counts and by every jump observable. Measured on ox1: with a far sentinel this
    observable would add ~24 raw nats to the silence penalty, against ~0.6 with zero -- and
    the counts alone already contribute 219.
    """
    if series_output:
        return V
    peak_idxs, _ = find_peaks(V, height=spike_min_thresh)
    if len(peak_idxs) < 2:
        return 0.0 * mb.max(V)          # zero, keeping the backend's type
    return (min_between_first_two_spikes(t, V, spike_min_thresh=spike_min_thresh)
            - steady_state_min(V))


@series_to_constant
def min_period(t, V, series_output=False, spike_min_thresh=None, distance=None):
    if series_output:
        return V
    # set distance = 5 to make sure it doesn't count a peak as two
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh, distance=distance)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 2:
        # there aren't enough peaks to calculate a period
        # so set the period_diff to the max time of the simulation
        period_min = t[-1] - t[0]
    else:
        # calculate the periods
        periods = [t[peak_idxs[II+1]] - t[peak_idxs[II]] for II in range(len(peak_idxs)-1)]
        # calculate the difference in time between max period and min period
        period_min = min(periods)

    return period_min

@series_to_constant
def first_period(t, V, series_output=False, spike_min_thresh=None, distance=None):
    if series_output:
        return V
    # set distance = 5 to make sure it doesn't count a peak as two
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh, distance=distance)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 1:
        # there aren't enough peaks to calculate a period
        # so set the period_diff to the max time of the simulation
        first_period = t[-1] - t[0]
    elif len(peak_idxs) < 2:
        # there aren't enough peaks to calculate a first period
        # so set the period_diff to the time to the first peak
        first_period = t[-1] - t[0]
    else:
        # calculate peaks without a threshold after first peak
        V_2 = V[peak_idxs[0]-2:] # first peak will be the same peak
        t_2 = t[peak_idxs[0]-2:] # first peak will be the same peak
        threshold_for_spike = min(V[peak_idxs[0]:peak_idxs[1]]) + 10 # setting this to try to ignore some noise
        peak_idxs_2, peak_properties_2 = find_peaks(V_2, height=threshold_for_spike, distance=distance)

        if len(peak_idxs_2) < 2:
            # there should have been another peak but for some reason it wasn't detected...
            first_period = t[-1] - t[0]
        else:
            # calculate the period
            first_period = t_2[peak_idxs_2[1]] - t[peak_idxs[0]]

    return first_period

@series_to_constant
def second_period(t, V, series_output=False, spike_min_thresh=None, distance=None):
    if series_output:
        return V
    # set distance = 5 to make sure it doesn't count a peak as two
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh, distance=distance)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 3:
        # there aren't enough peaks to calculate a period
        # so set the period_diff to the max time of the simulation
        second_period = t[-1] - t[0]
    else:
        # calculate peaks without a threshold after first peak
        V_2 = V[peak_idxs[1]-2:] # first peak of the next peak calc will be the same peak
        t_2 = t[peak_idxs[1]-2:] 
        threshold_for_spike = min(V[peak_idxs[1]:peak_idxs[2]]) + 10 # setting this to try to ignore some noise
        peak_idxs_2, peak_properties_2 = find_peaks(V_2, height=threshold_for_spike, distance=distance)

        if len(peak_idxs_2) < 2:
            # there should have been another peak but for some reason it wasn't detected...
            second_period = t[-1] - t[0]
        else:
            # calculate the period
            second_period = t_2[peak_idxs_2[1]] - t[peak_idxs[1]]

    return second_period

@series_to_constant
def E_A_ratio(t, x, T, series_output=False):
    if series_output:
        return x
    peak_idxs, peak_properties = find_peaks(x)
    if len(peak_idxs) < 1:
        # no peeak idxs found, return big value to make it a big cost
        return 100
    elif len(peak_idxs) < 2:
        # there is only one peak. E and A ontop of eachother. return large cost.
        return 10
    if np.isscalar(T):
        pass
    else:
        T = mb.mean(T) # take mean if this is T changing in time (T_wCont)

    if (t[peak_idxs[1]] - t[peak_idxs[0]] > 0.7* T) :
        # the peaks are too far apart. Probably because there is only one peak per heart beat.
        # return large value
        return 10

    # calculate with the first two peaks. This assumes that the E peak comes first # TODO make sure the E_peak comes first by passing in the
    E_A_ratio = x[peak_idxs[0]]/x[peak_idxs[1]]

    return E_A_ratio

# included by David Shaw
@series_to_constant
def peak_times(t, V, series_output=False):
    """
    returns all peak times
    """
    if series_output:
        return V
    peak_idxs, peak_properties = find_peaks(V)
    print(peak_idxs)
    if len(peak_idxs) == 0:
        return 99999999
    peaks = t[peak_idxs]
    return peaks

@differentiable
@series_to_constant
def mean_in_range(x, start_frac=0.0,end_frac=1.0, series_output=False):
    if series_output:
        return x
    else:
        start_idx = int(start_frac*(len(x)-1))
        end_idx = int(end_frac*(len(x)-1))
        range_values = x[start_idx:end_idx]
        return mb.mean(range_values)

@differentiable
@series_to_constant
def max_in_range(x, start_frac=0.0,end_frac=1.0,series_output=False):
    if series_output:
        return x
    else:
        start_idx = int(start_frac*(len(x)-1))
        end_idx = int(end_frac*(len(x)-1))
        range_values = x[start_idx:end_idx]
        return mb.max(range_values)

@series_to_constant
def max_first_half(x, series_output=False, start_frac=0.0, end_frac=0.5):
    print('max_first_half called')
    if series_output:
        return x
    else:
        start_idx = int(start_frac * (len(x) - 1))
        end_idx = int(end_frac * (len(x) - 1))
        range_values = x[start_idx:end_idx]
        return mb.min(range_values)

@series_to_constant
def V_plateau(t, V, series_output=False, spike_min_thresh=None, distance=None,
              dV_dt_thresh=10e3):
    """Mean voltage of the interspike plateau, after the AHP has recovered.

    For each action potential, take its peak and the trough that follows it. The plateau is
    taken to begin one peak-to-trough duration *after* the trough -- ``t_trough +
    (t_trough - t_peak)`` -- which skips the AHP and its recovery, and to end at the firing
    threshold of the next action potential, or at the end of the trace when there is none.
    The value is the mean over that window for the **last** action potential whose plateau
    start still falls inside it.

    This replaces a "steady state minimum". ``steady_state_min`` takes the minimum of the
    second half of the window, which on a repetitively firing trace is not a steady state at
    all -- it is simply a later AHP trough. Averaging between the AHP and the next threshold
    measures the interspike membrane potential the cell actually sits at, which is what a
    plateau is meant to be, and it is no longer a trough measurement in disguise.

    Thresholds come from :func:`_ap_thresholds`, the same walk :func:`mean_AP_threshold`
    reports, so the two observables are consistent by construction.

    With no action potentials there is no interspike interval; the mean of the second half of
    the trace is returned, which is the plateau of a silent trace.
    """
    if series_output:
        return V
    peak_idxs, _ = find_peaks(V, height=spike_min_thresh, distance=distance)
    fallback = mb.mean(V[len(V) // 2:])
    if len(peak_idxs) < 1:
        return fallback

    thresholds = _ap_thresholds(t, V, peak_idxs, dV_dt_thresh)
    window = None
    for i, peak_idx in enumerate(peak_idxs):
        next_peak = peak_idxs[i + 1] if i + 1 < len(peak_idxs) else len(V)
        trough_idx = peak_idx + int(np.argmin(V[peak_idx:next_peak]))
        start = t[trough_idx] + (t[trough_idx] - t[peak_idx])
        if i + 1 < len(peak_idxs) and thresholds[i + 1][0] is not None:
            end = t[thresholds[i + 1][0]]
        else:
            end = t[-1]
        if start < end:
            window = (start, end)          # keep the last AP that still has room
    if window is None:
        return fallback
    selected = (t >= window[0]) & (t <= window[1])
    if not np.any(selected):
        return fallback
    return mb.mean(V[selected])


def _ap_thresholds(t, V, peak_idxs, dV_dt_thresh):
    """``(index, voltage)`` where dV/dt first exceeds ``dV_dt_thresh`` ahead of each peak.

    The walk is the one :func:`mean_AP_threshold` has always used -- start three quarters of
    the way from the previous peak to this one, step forward until dV/dt crosses, then
    interpolate the voltage -- factored out so the plateau window can start and end at the
    same thresholds the threshold observable reports. ``(None, None)`` for a peak whose
    threshold runs off the end of the trace.
    """
    out = []
    prev_idx = 0
    for peak_idx in peak_idxs:
        current_idx = int((peak_idx + prev_idx) * 3 / 4)
        dV_dt = 0
        dV_dt_prev = 0
        while dV_dt < dV_dt_thresh and current_idx < len(t) - 1:
            dV_dt_prev = dV_dt
            dV_dt = (V[current_idx + 1] - V[current_idx]) / (t[current_idx + 1] - t[current_idx])
            current_idx += 1
        if current_idx < len(t) - 1:
            out.append((current_idx,
                        np.interp(dV_dt_thresh, [dV_dt_prev, dV_dt],
                                  [V[current_idx - 1], V[current_idx]])))
            prev_idx = peak_idx
        else:
            out.append((None, None))
    return out


@series_to_constant
def mean_AP_threshold(t, V, series_output=False, spike_min_thresh=None, distance=None, dV_dt_thresh=10e3):
    """
    This function calculates the mean action potential threshold
    using the peak detection algorithm from scipy.
    It finds the peaks in the voltage signal and then 
    moves back to pre AP (approximately) It then moves foreward until
    dV/dt is greater than dV_dt_thresh, default is 10 mV/ms (10e3 mV/s) from platkiewicz2010Threshold.

    # TODO this won't work with noise
    """
            
    if series_output:
        return V
    # set distance = 5 to make sure it doesn't count a peak as two
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh, distance=distance)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 1:
        # there are no peaks, so set value to mean of the voltage
        threshold = mb.mean(V)
    else:
        thresholds = [v for _, v in _ap_thresholds(t, V, peak_idxs, dV_dt_thresh)
                      if v is not None]

        if len(thresholds) == 0:
            # no thresholds found, exit
            print("no thresholds found, setting cost to large")
            threshold = 9999
        else:
            threshold = mb.mean(thresholds)

    return threshold

@series_to_constant
def mean_peak_to_trough_time(t, V, series_output=False, spike_min_thresh=None, distance=None):
    """
    This function calculates the time between the peak and trough of each action potential
    then takes the mean of them all
    """
            
    if series_output:
        return V
    # set distance = 5 to make sure it doesn't count a peak as two
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh, distance=distance)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 1:
        # there are no peaks, so set value to zero
        t_diff = 0
    else:
        t_diff_times = []
        for II in range(len(peak_idxs)):
            t_peak = t[peak_idxs[II]]

            next_peak_idx = peak_idxs[II + 1] if II + 1 < len(peak_idxs) else len(t) - 1
            trough_idx = np.argmin(V[peak_idxs[II]:next_peak_idx]) + peak_idxs[II]
            t_diff_times.append(t[trough_idx] - t_peak)
            
        t_diff = mb.mean(t_diff_times)

    return t_diff

@differentiable
@series_to_constant
def max_minus_min_divided_by_mean_in_range(x, start_frac=0.0, end_frac=1.0, series_output=False):
    # calculate the max minus min for the first max and min in a range.
    # for example: tidal volume = max(x) - min(x)
       
    start_idx = int(start_frac*(len(x)-1))
    end_idx = int(end_frac*(len(x)-1))
    range_values_max = mb.max(x[start_idx:end_idx])
    range_values_min = mb.min(x[start_idx:end_idx])
    range_values_mean = mb.mean(x[start_idx:end_idx])
    max_minus_min_divided_by_mean = (range_values_max - range_values_min)/range_values_mean

    if series_output:
        return x
    else:
        return max_minus_min_divided_by_mean

@differentiable
@series_to_constant
def max_minus_min_over_mean_in_range(x, start_frac=0.0, end_frac=1.0, series_output=False):
    return max_minus_min_divided_by_mean_in_range(
        x,
        start_frac=start_frac,
        end_frac=end_frac,
        series_output=series_output,
    )

@differentiable
def first_minus_second_over_third_in_range(first, second, third):
    return (first - second) / third

@differentiable
@series_to_constant
def max_minus_min_in_range(x, start_frac=0.0, end_frac=1.0, series_output=False):
    # calculate the max minus min for the first max and min in a range.
    # for example: tidal volume = max(x) - min(x)
       
    start_idx = int(start_frac*(len(x)-1))
    end_idx = int(end_frac*(len(x)-1))
    range_values_max = mb.max(x[start_idx:end_idx])
    range_values_min = mb.min(x[start_idx:end_idx])
    max_minus_min = range_values_max - range_values_min 

    if series_output:
        return x
    else:
        return max_minus_min

@differentiable
@series_to_constant
def max_minus_mean_in_range(x, start_frac=0.0, end_frac=1.0, series_output=False):
    # calculate the max minus min for the first max and min in a range.
    # for example: tidal volume = max(x) - min(x)
       
    start_idx = int(start_frac*(len(x)-1))
    end_idx = int(end_frac*(len(x)-1))
    range_values_max = mb.max(x[start_idx:end_idx])
    range_values_mean = mb.mean(x[start_idx:end_idx])
    max_minus_mean = range_values_max - range_values_mean 

    if series_output:
        return x
    else:
        return max_minus_mean

@differentiable
@series_to_constant
def mean_in_range_minus_initial(x, start_frac=0.8, end_frac=1.0, series_output=False):
    # calculate the mean in a range (normally at the end converged stated) minus the initial value in 
    # the subexperiment.
    # for example:
       
    start_idx = int(start_frac*(len(x)-1))
    end_idx = int(end_frac*(len(x)-1))
    range_values_mean = mb.mean(x[start_idx:end_idx])
    mean_minus_init = range_values_mean - x[0]

    if series_output:
        return x
    else:
        return mean_minus_init

@differentiable
@series_to_constant
def mean_in_range_fraction_change_from_initial(x, start_frac=0.8, end_frac=1.0, series_output=False):
    # calculate the mean in a range (normally at the end converged stated) minus the initial value in 
    # the subexperiment and get the percentage change.
    # for example:
       
    start_idx = int(start_frac*(len(x)-1))
    end_idx = int(end_frac*(len(x)-1))
    range_values_mean = mb.mean(x[start_idx:end_idx])
    mean_minus_init = range_values_mean - x[0]
    percentage_change = mean_minus_init / x[0]  # percentage change from initial value

    if series_output:
        # TODO for plotting should I output the percentage change or the mean minus initial?
        return x
    else:
        return percentage_change

@differentiable
@series_to_constant
def mean_in_range_fraction_change_from_initial_range(x, start_frac=0.8, end_frac=1.0, series_output=False, init_range_end_frac=0.1):
    # calculate the mean in a range (normally at the end converged stated) minus the initial value in 
    # the subexperiment and get the percentage change.
    # for example:
       
    start_idx = int(start_frac*(len(x)-1))
    end_idx = int(end_frac*(len(x)-1))
    end_init_idx = int(init_range_end_frac*(len(x)-1))
    range_values_mean = mb.mean(x[start_idx:end_idx])
    init_range_mean = mb.mean(x[:end_init_idx])
    mean_minus_init = range_values_mean - init_range_mean
    percentage_change = mean_minus_init / init_range_mean  # percentage change from initial value

    if series_output:
        # TODO for plotting should I output the percentage change or the mean minus initial?
        return x
    else:
        return percentage_change

@differentiable
@series_to_constant
def min_in_range(x, start_frac=0.0, end_frac=1.0, series_output=False):
    """
    Calculates the minimum value of the signal x in the window defined by start_frac and end_frac.
    start_frac and end_frac should be floats between 0 and 1, representing the fraction of the signal.
    """
    if series_output:
        return x
    
    start_idx = int(start_frac * (len(x) - 1))
    end_idx = int(end_frac * (len(x) - 1))
    range_values = x[start_idx:end_idx]

    return mb.min(range_values)


@series_to_constant
def calc_AHP_duration(t, V, baseline_voltage=None, series_output=False):
    """
    Calculates the duration of the afterhyperpolarization (AHP) as the time to 50% recovery.
    If there is more than one AP, returns the average AHP duration.
    baseline_voltage: If None, uses the mean of the first 5% of V as baseline.
    Returns np.nan if recovery is not reached.
    """
    if series_output:
        return V

    # Estimate baseline if not provided
    if baseline_voltage is None:
        baseline_voltage = mb.mean(V[:max(1, int(0.05 * len(V)))])

    # Find AP peaks (assume APs are positive peaks)
    peak_idxs, _ = find_peaks(V, height=baseline_voltage + 10)  # threshold can be adjusted

    if len(peak_idxs) < 1:
        return np.nan

    ahp_durations = []
    for peak_idx in peak_idxs:
        # Find minimum (hyperpolarization) after AP peak
        if peak_idx >= len(V) - 1:
            continue
        i_hyper = peak_idx + np.argmin(V[peak_idx:])
        V_hyper = V[i_hyper]

        # Calculate recovery level (50% recovery)
        recovery_level = baseline_voltage + 0.5 * (V_hyper - baseline_voltage)

        # Find time to recovery after hyperpolarization
        post_hyper = V[i_hyper:]
        try:
            i_recover = np.where(post_hyper >= recovery_level)[0][0] + i_hyper
            ahp_duration = t[i_recover] - t[i_hyper]
        except IndexError:
            ahp_duration = np.nan

        ahp_durations.append(ahp_duration)

    # Return average if multiple APs
    if len(ahp_durations) == 0:
        return np.nan
    else:
        return np.nanmean(ahp_durations)

@differentiable
@series_to_constant
def abs_diff_start_to_last_quarter(x, series_output=False):
    # TODO change to abs_diff_start to fraction
    if series_output:
        return x 
    else: 
        quarter_len = len(x) // 4
        first_value = x[0]
        last_quarter_values = x[-quarter_len:]
        return mb.abs(first_value - mb.mean(last_quarter_values))

##
## Below here are the organisational functions for building the operation functions dictionary
## They are not part of the public API
##

def register_user_operations(registry, backend):
    """
    Register user-defined observable operations. ``backend`` is the active
    MathBackend for this build (numpy or casadi); peak-based helpers remain
    NumPy/SciPy-only and are not marked @differentiable.

    Every top-level function defined in this module is registered except:
    private names (leading ``_``), ``series_to_constant``, ``register_user_operations``,
    ``ml_to_m3``, and names starting with ``RICRI_`` (parameter helpers).
    Imported callables are skipped via ``__module__`` checks.
    """
    global mb
    mb = backend
    g = globals()
    mod = __name__
    exclude = frozenset(
        {
            "series_to_constant",
            "register_user_operations",
            "ml_to_m3",
        }
    )
    for name, obj in g.items():
        if name.startswith("_") or name in exclude or name.startswith("RICRI_"):
            continue
        if not callable(obj) or isinstance(obj, type):
            continue
        if getattr(obj, "__module__", None) != mod:
            continue
        registry[name] = bind_backend(obj, backend)




@series_to_constant
def calculate_two_observable_difference(subtract_from=None, subtract_this=None,
                                        series_output=False):
    """``subtract_from - subtract_this``, where each names another data_item.

    The names say which way round the subtraction goes, which ``pred1``/``pred2``
    did not: nothing in those told you that the result was ``pred2 - pred1`` rather
    than the other way about, and getting it backwards is a sign error that looks
    like a plausible number.

    The two inputs are declared here rather than read out of ``**kwargs``: they are
    this function's arguments, so the signature is where they belong. Anything that
    introspects an operation -- ``get_operation_kwarg_spec``, and the GUI form built
    from it -- then sees them without having to read the body, and a misspelled key
    is refused by the usual ``operation_kwargs`` check instead of arriving here as a
    silently missing value.

    Both carry a default, which is what makes them *keyword* arguments rather than
    operands: a parameter with no default is filled positionally from the
    data_item's ``operands``, and these come from ``operation_kwargs``. The
    data_item therefore has no operands at all, and its ``operation_kwargs`` name
    the two items to difference:

        "operation": "calculate_two_observable_difference",
        "operands": [],
        "operation_kwargs": {"subtract_from": "peak prey, forced",
                             "subtract_this": "peak prey, unforced"}

    Renamed from ``pred1``/``pred2``: an obs_data using the old keys is refused,
    naming them, rather than quietly computing anything. ``pred2`` is the value
    subtracted *from*, so it becomes ``subtract_from``, and ``pred1`` becomes
    ``subtract_this`` -- swapping them would flip the sign of the result.
    """
    if subtract_from is None:
        raise RuntimeError(
            "calculate_two_observable_difference: 'subtract_from' was not supplied. It "
            "names the data_item to subtract from; set it in this data_item's "
            "operation_kwargs. (It was called 'pred2' before.)")
    if subtract_this is None:
        raise RuntimeError(
            "calculate_two_observable_difference: 'subtract_this' was not supplied. It "
            "names the data_item to subtract; set it in this data_item's "
            "operation_kwargs. (It was called 'pred1' before.)")

    return subtract_from - subtract_this



