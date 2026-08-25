"""How the fitted emulator gets pickled, and what happens when joblib cannot.

Training pays for every simulation *before* it saves. So a model joblib cannot
pickle does not cost a save, it costs the whole run -- which is what issue #468
reports: `joblib.dump` raising ``cannot pickle '_abc._abc_data' object`` on an
emulator holding an uninitialised C-extension descriptor.

That exact object is autoemulate's and is not reproducible on demand, so what is
reproduced here is the *mechanism*: an object pickle refuses and dill accepts.
A closure is the simplest one, and it fails in the same place for the same
reason -- pickle cannot name the thing it is being asked to store.
"""
import json
import os

import numpy as np
import pytest

from libcuflynx.emulators.emulator_bundle import (
    _AUTO_ORDER,
    DEFAULT_SERIALISER,
    METADATA_FILE,
    MODEL_FILE,
    SERIALISERS,
    EmulatorBundle,
    _load_model,
    _save_model,
)

pytestmark = pytest.mark.unit

dill = pytest.importorskip("dill")


class LinearStub:
    """A trivial emulator: y = x @ w. Picklable by anything."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, x):
        return np.asarray(x, dtype=float) @ self.weights


class ClosureStub(LinearStub):
    """The same emulator, holding something pickle cannot name.

    Stands in for the ``_abc._abc_data`` of #468: a live object with no
    importable path to it, which pickle refuses and dill takes apart.
    """

    def __init__(self, weights):
        super().__init__(weights)
        scale = float(weights[0][0])
        self.transform = lambda value: value * scale


def _meta(serialiser=None):
    meta = {
        'param_entry_labels': ['alpha'],
        'param_mins': [0.0],
        'param_maxs': [1.0],
        'feature_labels': ['max (max comp/x)'],
        'feature_r2': [0.99],
        'x_scale': {'shift': [0.0], 'span': [1.0]},
        'y_scale': {'shift': [0.0], 'span': [1.0]},
        'fingerprint': {'inputs_sha256': 'abc123'},
    }
    if serialiser is not None:
        meta['settings'] = {'model_serialiser': serialiser}
    return meta


def _model_path(directory):
    return os.path.join(directory, MODEL_FILE)


# ---------------------------------------------------------------------------
# The failure this exists for
# ---------------------------------------------------------------------------
def test_joblib_really_cannot_pickle_the_stand_in():
    """The premise. Without this the fallback tests could pass by accident."""
    import joblib

    with pytest.raises(Exception) as exc:
        joblib.dump(ClosureStub([[2.0]]), os.devnull)
    assert 'pickle' in str(exc.value).lower()


def test_auto_falls_back_rather_than_losing_the_run(tmp_path, capsys):
    used = _save_model(ClosureStub([[2.0]]), _model_path(tmp_path), serialiser='auto')
    assert used != 'joblib'
    assert used in _AUTO_ORDER
    assert os.path.isfile(f'{_model_path(tmp_path)}.joblib')
    # Silently switching container would be worse than failing: the file now
    # needs that library wherever it is read.
    assert used in capsys.readouterr().out


def test_auto_uses_joblib_when_joblib_works(tmp_path):
    """The fallback is a fallback. joblib is what autoemulate itself reads."""
    assert _save_model(LinearStub([[2.0]]), _model_path(tmp_path), serialiser='auto') == 'joblib'


def test_a_failed_joblib_attempt_leaves_no_half_written_file(tmp_path):
    """A truncated file would be loaded in preference to nothing at all."""
    path = f'{_model_path(tmp_path)}.joblib'
    _save_model(ClosureStub([[2.0]]), _model_path(tmp_path), serialiser='auto')
    with open(path, 'rb') as file:
        assert dill.load(file).predict([[1.0]])[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------
def test_the_default_is_auto():
    assert DEFAULT_SERIALISER == 'auto'
    assert set(SERIALISERS) == {'auto', *_AUTO_ORDER}


def test_auto_tries_joblib_first_and_dill_last():
    """Order is measured, not assumed. joblib is what autoemulate itself writes;
    dill is last because on autoemulate 2.1.2 it *fails* where joblib succeeds --
    a torch-backed emulator holds a PyCapsule it recurses on -- so promoting it,
    as issue #468 proposes, would break the common case to fix the rare one."""
    assert _AUTO_ORDER[0] == 'joblib'
    assert _AUTO_ORDER[-1] == 'dill'
    assert 'cloudpickle' in _AUTO_ORDER


def test_naming_dill_uses_dill_without_trying_joblib_first(tmp_path):
    assert _save_model(LinearStub([[2.0]]), _model_path(tmp_path), serialiser='dill') == 'dill'


def test_naming_joblib_does_not_fall_back(tmp_path):
    """Someone who has pinned joblib wants to hear about it, not get a dill file."""
    with pytest.raises(Exception) as exc:
        _save_model(ClosureStub([[2.0]]), _model_path(tmp_path), serialiser='joblib')
    assert 'pickle' in str(exc.value).lower()


def test_an_unknown_serialiser_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match='model_serialiser'):
        _save_model(LinearStub([[2.0]]), _model_path(tmp_path), serialiser='marshal')


@pytest.mark.parametrize('name', ['cloudpickle', 'dill'])
def test_each_fallback_container_round_trips_what_joblib_refuses(name, tmp_path):
    """The premise of the fallback, checked per container rather than assumed."""
    pytest.importorskip(name)
    assert _save_model(ClosureStub([[2.0]]), _model_path(tmp_path), serialiser=name) == name
    back = _load_model(_model_path(tmp_path), serialiser=name)
    assert back.predict([[1.0]])[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Round trip through the bundle
# ---------------------------------------------------------------------------
def test_a_bundle_records_which_serialiser_wrote_it(tmp_path):
    """Not inferred on the way back in: the containers are not distinguishable
    from the bytes."""
    EmulatorBundle(ClosureStub([[2.0]]), _meta()).save(str(tmp_path))
    with open(tmp_path / METADATA_FILE) as file:
        recorded = json.load(file)['model_serialiser']
    assert recorded in _AUTO_ORDER and recorded != 'joblib'


def test_a_fallback_saved_bundle_loads_back(tmp_path):
    EmulatorBundle(ClosureStub([[2.0]]), _meta()).save(str(tmp_path))
    reloaded = EmulatorBundle.load(str(tmp_path))
    assert reloaded.predict([0.5])[0] == pytest.approx(1.0)


def test_a_joblib_saved_bundle_still_loads_back(tmp_path):
    EmulatorBundle(LinearStub([[2.0]]), _meta('joblib')).save(str(tmp_path))
    assert EmulatorBundle.load(str(tmp_path)).predict([0.5])[0] == pytest.approx(1.0)


def test_the_setting_reaches_save_from_the_bundles_own_settings(tmp_path):
    """emulator_settings travels in the metadata, the way min_r2 does -- by load
    time the caller may be a calibration run that never saw those settings."""
    EmulatorBundle(LinearStub([[2.0]]), _meta('dill')).save(str(tmp_path))
    with open(tmp_path / METADATA_FILE) as file:
        assert json.load(file)['model_serialiser'] == 'dill'


def test_a_bundle_saved_before_the_choice_existed_still_loads(tmp_path):
    """No recorded serialiser: both containers get tried."""
    EmulatorBundle(LinearStub([[2.0]]), _meta()).save(str(tmp_path))
    path = tmp_path / METADATA_FILE
    meta = json.loads(path.read_text())
    del meta['model_serialiser']
    path.write_text(json.dumps(meta))
    assert EmulatorBundle.load(str(tmp_path)).predict([0.5])[0] == pytest.approx(1.0)


def test_a_fallback_file_loads_even_if_the_metadata_claims_joblib(tmp_path):
    """The recorded name is a preference, not a promise -- a bundle hand-edited
    or copied between studies must not become unreadable."""
    EmulatorBundle(ClosureStub([[2.0]]), _meta()).save(str(tmp_path))
    path = tmp_path / METADATA_FILE
    meta = json.loads(path.read_text())
    meta['model_serialiser'] = 'joblib'
    path.write_text(json.dumps(meta))
    assert EmulatorBundle.load(str(tmp_path)).predict([0.5])[0] == pytest.approx(1.0)


def test_a_missing_model_file_is_reported_as_such(tmp_path):
    with pytest.raises(FileNotFoundError, match='is missing'):
        _load_model(_model_path(tmp_path))
