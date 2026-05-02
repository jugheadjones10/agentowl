import os
import sys
import importlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classes.helper import StateTransitionTriplet, set_global_constants  # noqa: E402
from data.atari import load_atari_observations  # noqa: E402
from learners.synthesizer import ActionSynthesizer, NoInteractPassiveMovementSynthesizer  # noqa: E402

DEFAULT_TASK = "MontezumaRevenge"
DEFAULT_OBS_SUFFIX = "_basic17"


class FakeAsyncLLM:
    """Deterministic async LLM for learning tests.

    It records prompts so tests can inspect the exact data sent into each LLM
    stage without spending API calls or depending on nondeterministic output.
    """

    is_fake = True

    def __init__(self):
        self.calls = []

    async def aprompt(self, prompts, temperature=0, seed=None):
        outputs = [self._response_for(prompt) for prompt in prompts]
        self.calls.append({
            "prompts": prompts,
            "temperature": temperature,
            "seed": seed,
            "outputs": outputs,
        })
        return outputs

    def _response_for(self, prompt):
        obj_type = self._obj_type_for_prompt(prompt)
        if "We want to synthesize python functions" in prompt:
            return f"""1.
```python
def alter_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    # Since no {obj_type} objects are created, we simply return the original obj_list without any modifications.
    return obj_list
```

2.
```python
def alter_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}')
    for {obj_type}_obj in {obj_type}_objs:
        # Set the deleted attribute to 0, indicating the object is not deleted.
        {obj_type}_obj.deleted = RandomValues([0])
    return obj_list
```

3.
```python
def alter_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}')
    for {obj_type}_obj in {obj_type}_objs:
        # Set the x-axis velocity to 0.
        {obj_type}_obj.velocity_x = RandomValues([0])
    return obj_list
```

4.
```python
def alter_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}')
    for {obj_type}_obj in {obj_type}_objs:
        # Set the y-axis velocity to 0.
        {obj_type}_obj.velocity_y = RandomValues([0])
    return obj_list
```"""

        return (
            f"1. No {obj_type} objects are created.\n"
            f"2. The {obj_type} objects are not deleted.\n"
            f"3. The {obj_type} objects set their x-axis velocity to +0.\n"
            f"4. The {obj_type} objects set their y-axis velocity to +0.\n")

    def _obj_type_for_prompt(self, prompt):
        patterns = [
            r"possible behaviors of ([a-zA-Z_]+) objects",
            r"reasons of the ([a-zA-Z_]+) objects",
            r"def alter_([a-zA-Z_]+)_objects",
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt)
            if match:
                return match.group(1)
        return "object"


class RecordingLLM:
    """Wrap a real LLM and keep raw prompts/outputs for inspection."""

    is_fake = False

    def __init__(self, llm):
        self.llm = llm
        self.calls = []

    async def aprompt(self, prompts, temperature=0, seed=None):
        outputs = await self.llm.aprompt(prompts,
                                         temperature=temperature,
                                         seed=seed)
        self.calls.append({
            "prompts": prompts,
            "temperature": temperature,
            "seed": seed,
            "outputs": outputs,
        })
        return outputs

    def __getattr__(self, name):
        return getattr(self.llm, name)


def pytest_addoption(parser):
    parser.addoption(
        "--real-llm",
        action="store_true",
        default=False,
        help="Run opt-in learning tests that call the configured real LLM.",
    )


@pytest.fixture
def learning_config():
    return OmegaConf.create({
        "task": DEFAULT_TASK,
        "obs_suffix": DEFAULT_OBS_SUFFIX,
        "obs_index": 0,
        "obs_index_length": 8,
        "seed": 0,
        "provider": "openrouter",
        "use_memory": False,
        "database_path": "debug_completions_synthesizer_learning.db",
        "synthesizer": {
            "synth_window": 1,
        },
    })


@pytest.fixture
def saved_observations(learning_config):
    set_global_constants(learning_config.task)
    identifier = learning_config.task.replace("Alt",
                                              "") + learning_config.obs_suffix
    path = Path("saved_data") / f"obs_{identifier}.pickle"
    if not path.exists():
        pytest.skip(f"Missing saved observation fixture: {path}")
    return load_atari_observations(identifier)


@pytest.fixture
def saved_observation_slice(saved_observations, learning_config):
    observations, actions, game_states = saved_observations
    start = learning_config.obs_index
    stop = start + learning_config.obs_index_length
    return SimpleNamespace(
        observations=observations[start:stop + 1],
        actions=actions[start:stop],
        game_states=game_states[start:stop + 1],
    )


@pytest.fixture
def transition_slice(saved_observation_slice):
    return [
        StateTransitionTriplet(
            saved_observation_slice.observations[i],
            saved_observation_slice.actions[i],
            saved_observation_slice.observations[i + 1],
            input_game_state=saved_observation_slice.game_states[i],
            output_game_state=saved_observation_slice.game_states[i + 1],
        ) for i in range(len(saved_observation_slice.actions))
    ]


@pytest.fixture
def no_interact_passive_movement_case(transition_slice, learning_config,
                                      learning_llm):
    synth = NoInteractPassiveMovementSynthesizer(
        learning_config,
        "player",
        learning_llm,
    )
    transition = transition_slice[0]

    return SimpleNamespace(
        obj_type="player",
        transition=transition,
        transitions=transition_slice,
        llm=learning_llm,
        synth=synth,
    )

@pytest.fixture
def synthesizer_obj(transition_slice, learning_config, learning_llm):
    synth = ActionSynthesizer(
        learning_config,
        "player",
        learning_llm,
    )

    return SimpleNamespace(
        obj_type="player",
        transitions=transition_slice,
        llm=learning_llm,
        synth=synth,
    )


@pytest.fixture
def real_llm_enabled(pytestconfig):
    return pytestconfig.getoption("--real-llm") or os.getenv(
        "AGENTOWL_REAL_LLM") == "1"


@pytest.fixture
def learning_llm(real_llm_enabled, learning_config):
    if not real_llm_enabled:
        return FakeAsyncLLM()
    try:
        openai_hf_interface = importlib.import_module("openai_hf_interface")
        openai_hf_interface.choose_provider(learning_config.provider)
        create_llm = openai_hf_interface.create_llm

        model_name = ("gpt-4o-2024-08-06" if learning_config.provider
                      == "openai" else "openai/gpt-4o-2024-08-06")
        llm = create_llm(model_name)
        llm.setup_cache("disk", database_path=learning_config.database_path)
        llm.set_default_kwargs({"timeout": 60})
        return RecordingLLM(llm)
    except Exception as exc:
        pytest.skip(f"Could not initialize real LLM: {exc}")
