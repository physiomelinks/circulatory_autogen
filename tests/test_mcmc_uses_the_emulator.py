"""``use_emulator: true`` has to reach the MCMC stage, not just the calibration.

The calibration engine is built by ``CVS0DParamID.init_from_dict``, which reads
``use_emulator`` / ``emulator_dir`` / ``emulator_settings`` out of the config. The
MCMC engine in ``run_param_id`` is constructed by hand, and the emulator arguments
were simply not in the call -- so a run with ``use_emulator: true`` accelerated the
calibration and then sampled against the solver.

That is the one stage where the difference is unaffordable: a chain is tens of
thousands of evaluations, and the symptom is not an error but a run that never
finishes. Nothing failed, so nothing caught it.

The engines are stubbed here. The behaviour under test is entirely "what was this
constructor called with", and building a real one compiles a model.
"""
import pytest

from libcuflynx.scripts import param_id_run_script as stage


class FakeEngine:
    """Records its construction, and does nothing expensive."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.output_dir = kwargs.get("param_id_output_dir", ".")
        self.param_id = object()
        FakeEngine.instances.append(self)

    # The surface run_param_id touches.
    def run(self):
        pass

    def get_best_param_vals(self):
        return [1.0, 2.0]

    def set_best_param_vals(self, values):
        self.best_param_vals = values

    def set_ground_truth_data(self, obs):
        pass

    def set_params_for_id(self, params):
        pass

    def run_mcmc(self):
        self.ran_mcmc = True


EMULATOR_DIR = "/tmp/an-emulator-that-was-trained"


def _config(**over):
    config = {
        "DEBUG": False,
        "model_path": "model.cellml",
        "model_type": "cellml",
        "param_id_method": "genetic_algorithm",
        "file_prefix": "demo",
        "params_for_id_path": "params.csv",
        "param_id_obs_path": "obs.json",
        "sim_time": 2.0,
        "pre_time": 1.0,
        "solver_info": {"solver": "CVODE_myokit"},
        "dt": 0.01,
        "optimiser_options": {"num_calls_to_function": 10},
        "resources_dir": "resources",
        "param_id_output_dir": "out",
        "do_ad": False,
        "do_uq": True,
        "UQ_options": {"method": "mcmc", "library": "emcee"},
        "debug_UQ_options": {"method": "mcmc", "library": "emcee"},
        "do_ia": False,
        "use_emulator": True,
        "emulator_settings": {"emulator_dir": EMULATOR_DIR, "min_r2": 0.9},
    }
    config.update(over)
    return config


@pytest.fixture
def engines(monkeypatch):
    FakeEngine.instances = []
    monkeypatch.setattr(stage, "CVS0DParamID", FakeEngine)
    monkeypatch.setattr(
        stage, "ensure_mle_cost_type_for_bayesian_inner", lambda *a, **k: None)
    # The parser would validate paths and read files; the config is the input here.
    monkeypatch.setattr(
        stage.YamlFileParser, "parse_user_inputs_file",
        lambda self, inp, **kwargs: inp)
    monkeypatch.setattr(stage.os.path, "exists", lambda path: False)
    return FakeEngine.instances


@pytest.mark.unit
def test_the_mcmc_engine_is_given_the_emulator(engines):
    stage.run_param_id(_config())

    assert len(engines) == 2, "expected a calibration engine and an MCMC engine"
    mcmc = engines[1]
    assert mcmc.args[3] is True, "the second engine should be the mcmc_instead one"
    assert mcmc.kwargs.get("use_emulator") is True
    assert mcmc.kwargs.get("emulator_dir") == EMULATOR_DIR
    assert mcmc.kwargs.get("emulator_settings") == {
        "emulator_dir": EMULATOR_DIR, "min_r2": 0.9}


@pytest.mark.unit
def test_the_mcmc_actually_ran(engines):
    stage.run_param_id(_config())

    assert getattr(engines[1], "ran_mcmc", False)


@pytest.mark.unit
def test_no_emulator_means_no_emulator_dir(engines):
    """Without use_emulator nothing is resolved -- importing the trainer pulls
    autoemulate, which is an optional extra."""
    stage.run_param_id(_config(use_emulator=False))

    mcmc = engines[1]
    assert mcmc.kwargs.get("use_emulator") is False
    assert mcmc.kwargs.get("emulator_dir") is None


@pytest.mark.unit
def test_the_calibration_still_gets_it_too(engines):
    """init_from_dict already handled the calibration engine; this must not have
    changed that."""
    stage.run_param_id(_config())

    # The calibration engine is built through init_from_dict, so it is not one of
    # the recorded direct constructions -- only the MCMC one is.
    assert len(engines) == 2
    assert engines[0].args[3] is False
