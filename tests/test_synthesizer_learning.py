"""Learning harness for stepping through synthesizer behavior.

Run with:
    uv run pytest tests/test_synthesizer_learning.py -s

Opt into real LLM calls for the same tests with:
    AGENTOWL_REAL_LLM=1 uv run pytest tests/test_synthesizer_learning.py -s
"""

import asyncio
import json

from learners.utils import partial_format
from prompts.synthesizer import interpret_no_int_prompt


def _player_transition_repr(transition):
    input_player = transition.input_state.get_objs_by_obj_type('player')[0]
    output_player = transition.output_state.get_objs_by_obj_type('player')[0]
    return {
        'event': transition.event,
        'input_game_state': transition.input_game_state.name,
        'output_game_state': transition.output_game_state.name,
        'input_state': {
            'player_repr_lines': repr(input_player).splitlines(),
        },
        'output_state': {
            'player_repr_lines': repr(output_player).splitlines(),
        },
    }


def _non_empty_lines_between(text, start_marker, end_marker):
    section = text.split(start_marker, 1)[1].split(end_marker, 1)[0]
    return [line for line in section.splitlines() if line]


def _interpret_prompt_repr(prompt):
    return {
        'input_lines': _non_empty_lines_between(
            prompt,
            'Input list of objects:\n',
            '\nOutput list of object changes:',
        ),
        'effect_lines': _non_empty_lines_between(
            prompt,
            'Output list of object changes:\n',
            '\nPlease follow these rules for your output:',
        ),
    }


def _code_prompt_repr(prompt):
    return {
        'observation_lines': _non_empty_lines_between(
            prompt,
            'We observe that the possible behaviors of player objects include\n',
            '\nWe want to synthesize python functions',
        ),
        'function_header': 'def alter_player_objects(obj_list: ObjList, _) -> ObjList:',
    }


def _llm_call_repr(call, prompt_repr):
    return {
        'prompt_count': len(call['prompts']),
        'prompt': prompt_repr(call['prompts'][0]),
        'temperature': call['temperature'],
        'seed': call['seed'],
        'output_lines': call['outputs'][0].splitlines(),
    }


def _without_numbering(line):
    return line.split('. ', 1)[1] if '. ' in line else line


def _without_bullet(line):
    return line.removeprefix('- ')


def _expected_no_interact_passive_player_rules():
    return [
        'def alter_player_objects(obj_list: ObjList, _) -> ObjList:\n'
        '    # Since no player objects are created, we simply return the original obj_list without any modifications.\n'
        '    return obj_list',
        'def alter_player_objects(obj_list: ObjList, _) -> ObjList:\n'
        "    player_objs = obj_list.get_objs_by_obj_type('player')\n"
        '    for player_obj in player_objs:\n'
        '        # Set the deleted attribute to 0, indicating the object is not deleted.\n'
        '        player_obj.deleted = RandomValues([0])\n'
        '    return obj_list',
        'def alter_player_objects(obj_list: ObjList, _) -> ObjList:\n'
        "    player_objs = obj_list.get_objs_by_obj_type('player')\n"
        '    for player_obj in player_objs:\n'
        '        # Set the x-axis velocity to 0.\n'
        '        player_obj.velocity_x = RandomValues([0])\n'
        '    return obj_list',
        'def alter_player_objects(obj_list: ObjList, _) -> ObjList:\n'
        "    player_objs = obj_list.get_objs_by_obj_type('player')\n"
        '    for player_obj in player_objs:\n'
        '        # Set the y-axis velocity to 0.\n'
        '        player_obj.velocity_y = RandomValues([0])\n'
        '    return obj_list',
    ]


def test_saved_observation_fixture_matches_run_slice(
    saved_observation_slice,
    transition_slice,
    learning_config,
):
    """Show that the harness mirrors `run.py`'s observation slicing.

    `run.py` creates N transitions from N actions and N + 1 observations. This
    test protects that shape so later synthesizer tests are using real data in
    the same form the world-model learner receives.
    """
    assert len(saved_observation_slice.observations) == (
        learning_config.obs_index_length + 1
    )
    assert (
        len(saved_observation_slice.actions)
        == learning_config.obs_index_length
    )
    assert len(saved_observation_slice.game_states) == (
        learning_config.obs_index_length + 1
    )
    assert len(transition_slice) == learning_config.obs_index_length
    assert (
        transition_slice[0].input_state
        is saved_observation_slice.observations[0]
    )
    assert transition_slice[0].event == saved_observation_slice.actions[0]
    assert transition_slice[0].output_state is not None

    # When run with `-s`, this prints a compact view of the first real
    # transition so it can be inspected while stepping through the test.
    print(json.dumps(_player_transition_repr(transition_slice[0]), indent=2))


def test_no_interact_passive_movement_effects_from_real_transition(
    no_interact_passive_movement_case,
):
    """Inspect the raw object-change facts extracted before any LLM call.

    `_get_natural_language_effects` turns a real `StateTransitionTriplet` into
    bullet-worthy facts such as velocity changes, deletion, or creation.
    """
    case = no_interact_passive_movement_case
    synth = case.synth
    transition = case.transition

    expected_transition_repr = {
        'event': 'NOOP',
        'input_game_state': 'RESTART',
        'output_game_state': 'NORMAL',
        'input_state': {
            'player_repr_lines': [
                'player object at (x=76, y=73, falling_time=0, velocity x=0, y=0)'
            ],
        },
        'output_state': {
            'player_repr_lines': [
                'player object at (x=76, y=73, falling_time=0, velocity x=0, y=0)'
            ],
        },
    }

    transition_repr = _player_transition_repr(transition)
    effects = synth._get_natural_language_effects(transition)
    cached_effects = synth._get_natural_language_effects(transition)

    expected_effects = [
        'The player object (id = 0) sets x-axis velocity to +0',
        'The player object (id = 0) sets y-axis velocity to +0',
    ]

    assert transition_repr == expected_transition_repr
    assert effects == expected_effects
    # The method caches by transition object so repeated calls reuse the same
    # list. This matters because later stages may call it multiple times.
    assert cached_effects is effects
    print(
        json.dumps(
            {
                'obj_type': case.obj_type,
                'transition': transition_repr,
                'effects': effects,
            },
            indent=2,
        )
    )


def test_no_interact_passive_movement_natural_language_observation_stage(
    no_interact_passive_movement_case,
):
    """Run only stage 1 of synthesis: facts -> natural-language observations.

    The `learning_llm` fixture decides whether this uses the deterministic fake
    LLM or the opt-in real LLM. Either way, the synthesizer sees the same
    `aprompt(...)` interface.
    """
    case = no_interact_passive_movement_case
    synth = case.synth
    prompt = partial_format(interpret_no_int_prompt, obj_type=case.obj_type)

    observations = asyncio.run(
        synth._a_get_natural_language_observations(
            [case.transition],
            prompt,
        )
    )
    call = case.llm.calls[0]
    stage_repr = {
        'transition': _player_transition_repr(case.transition),
        'llm_call': _llm_call_repr(call, _interpret_prompt_repr),
        'observations': observations,
    }
    expected_observations = [
        'No player objects are created.',
        'The player objects are not deleted.',
        'The player objects set their x-axis velocity to +0.',
        'The player objects set their y-axis velocity to +0.',
    ]
    expected_stage_repr = {
        'transition': {
            'event': 'NOOP',
            'input_game_state': 'RESTART',
            'output_game_state': 'NORMAL',
            'input_state': {
                'player_repr_lines': [
                    'player object at (x=76, y=73, falling_time=0, velocity x=0, y=0)'
                ],
            },
            'output_state': {
                'player_repr_lines': [
                    'player object at (x=76, y=73, falling_time=0, velocity x=0, y=0)'
                ],
            },
        },
        'llm_call': {
            'prompt_count': 1,
            'prompt': {
                'input_lines': ['player object (id = 0),'],
                'effect_lines': [
                    '- The player object (id = 0) sets x-axis velocity to +0',
                    '- The player object (id = 0) sets y-axis velocity to +0',
                ],
            },
            'temperature': 0,
            'seed': 0,
            'output_lines': [
                '1. No player objects are created.',
                '2. The player objects are not deleted.',
                '3. The player objects set their x-axis velocity to +0.',
                '4. The player objects set their y-axis velocity to +0.',
            ],
        },
        'observations': expected_observations,
    }

    assert isinstance(observations, list)
    assert len(case.llm.calls) == 1
    assert stage_repr['transition'] == expected_stage_repr['transition']
    assert (
        stage_repr['llm_call']['prompt']
        == expected_stage_repr['llm_call']['prompt']
    )
    assert (
        stage_repr['llm_call']['prompt_count']
        == expected_stage_repr['llm_call']['prompt_count']
    )
    assert (
        stage_repr['llm_call']['temperature']
        == expected_stage_repr['llm_call']['temperature']
    )
    assert (
        stage_repr['llm_call']['seed']
        == expected_stage_repr['llm_call']['seed']
    )
    assert set(observations) == set(expected_observations)
    assert {
        _without_numbering(line)
        for line in stage_repr['llm_call']['output_lines']
    } == set(expected_observations)
    print(json.dumps(stage_repr, indent=2))


def test_no_interact_passive_movement_synthesizer_end_to_end(
    no_interact_passive_movement_case,
):
    """Run the full two-stage passive synthesizer.

    This covers the component path:
    real transitions -> interpretation prompt -> observations -> code prompt ->
    parsed Python rule strings.
    """
    case = no_interact_passive_movement_case
    rules = asyncio.run(case.synth.a_synthesize(case.transitions))
    assert len(case.llm.calls) == 2
    interpret_llm_call = _llm_call_repr(
        case.llm.calls[0], _interpret_prompt_repr
    )
    code_llm_call = _llm_call_repr(case.llm.calls[1], _code_prompt_repr)
    code_prompt = code_llm_call['prompt']
    synthesis_repr = {
        'input_transition': _player_transition_repr(case.transition),
        'interpret_llm_call': interpret_llm_call,
        'code_llm_call': code_llm_call,
        'rules': rules,
    }
    synthesis_comparison_repr = {
        'input_transition': synthesis_repr['input_transition'],
        'interpret_llm_call': {
            'prompt_count': interpret_llm_call['prompt_count'],
            'prompt': interpret_llm_call['prompt'],
            'temperature': interpret_llm_call['temperature'],
            'seed': interpret_llm_call['seed'],
            'output_observations': sorted(
                _without_numbering(line)
                for line in interpret_llm_call['output_lines']
            ),
        },
        'code_llm_call': {
            'prompt_count': code_llm_call['prompt_count'],
            'prompt': {
                'observation_lines': sorted(
                    _without_bullet(line)
                    for line in code_prompt['observation_lines']
                ),
                'function_header': code_prompt['function_header'],
            },
            'temperature': code_llm_call['temperature'],
            'seed': code_llm_call['seed'],
        },
        'rules': rules,
    }
    expected_observations = [
        'No player objects are created.',
        'The player objects are not deleted.',
        'The player objects set their x-axis velocity to +0.',
        'The player objects set their y-axis velocity to +0.',
    ]
    expected_synthesis_repr = {
        'input_transition': {
            'event': 'NOOP',
            'input_game_state': 'RESTART',
            'output_game_state': 'NORMAL',
            'input_state': {
                'player_repr_lines': [
                    'player object at (x=76, y=73, falling_time=0, velocity x=0, y=0)'
                ],
            },
            'output_state': {
                'player_repr_lines': [
                    'player object at (x=76, y=73, falling_time=0, velocity x=0, y=0)'
                ],
            },
        },
        'interpret_llm_call': {
            'prompt_count': 1,
            'prompt': {
                'input_lines': ['player object (id = 0),'],
                'effect_lines': [
                    '- The player object (id = 0) sets x-axis velocity to +0',
                    '- The player object (id = 0) sets y-axis velocity to +0',
                ],
            },
            'temperature': 0,
            'seed': 0,
            'output_observations': sorted(expected_observations),
        },
        'code_llm_call': {
            'prompt_count': 1,
            'prompt': {
                'observation_lines': sorted(expected_observations),
                'function_header': 'def alter_player_objects(obj_list: ObjList, _) -> ObjList:',
            },
            'temperature': 0,
            'seed': 0,
        },
        'rules': _expected_no_interact_passive_player_rules(),
    }

    assert isinstance(rules, list)
    assert synthesis_comparison_repr == expected_synthesis_repr
    print(json.dumps(synthesis_repr, indent=2))
