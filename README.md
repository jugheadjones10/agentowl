# [PoE-World: Compositional World Modeling with Products of Programmatic Experts (NeurIPS 2025 Spotlight)](https://arxiv.org/abs/2505.10819)

By [Wasu Top Piriyakulkij](https://www.cs.cornell.edu/~wp237/), [Yichao Liang](https://yichao-liang.github.io/), [Hao Tang](https://haotang1995.github.io/), [Adrian Weller](https://mlg.eng.cam.ac.uk/adrian/), [Marta Kryven](https://marta-kryven.github.io/), [Kevin Ellis](https://www.cs.cornell.edu/~ellisk/)

This GitHub repo ([agentowl](https://github.com/jugheadjones10/agentowl)) is a fork of the upstream project [topwasu/poe-world](https://github.com/topwasu/poe-world).

[![deploy](https://img.shields.io/badge/Project_Page%20%20-8A2BE2)](https://topwasu.github.io/poe-world) [![arXiv](https://img.shields.io/badge/arXiv-2401.02739-red.svg)](https://arxiv.org/abs/2505.10819)


We introduce a novel program synthesis approach to output world models of complex, non-gridworld domains by representing world models as products of programmatic experts.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`). Python **3.10** is pinned via `.python-version`.

Install uv (if needed): see [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

Clone and enter the repo:
```
git clone https://github.com/jugheadjones10/agentowl.git
cd agentowl
```

Initialize submodules (required before install—local packages `openai-hf-interface` and `OC_Atari` are linked from `pyproject.toml`):
```
git submodule update --init --recursive
```

Create the virtualenv and install all runtime dependencies (including editable `openai-hf-interface` and `ocatari`, and Atari-enabled `gymnasium`):
```
uv sync
```

Optional dev tools (formatters, coverage, etc.):
```
uv sync --group dev
```

Julia is still required on your machine for `juliacall` / `juliapkg` (same as upstream).

**OpenAI API key:** create `openai-hf-interface/secrets.json` with `"openai_api_key": "<your key>"`. See [openai-hf-interface](https://github.com/topwasu/openai-hf-interface) for details.

## Running

Use `uv run` so commands use the project environment (or activate `.venv` and call `python` as usual).

Running PoE-World
```
uv run python make_observations.py task=Pong # choose task from [Pong, PongAlt, MontezumaRevenge, MontezumaRevengeAlt]
uv run python run.py --config-name=pong_agent # choose config-name from [pong_agent, pong_alt_agent, montezuma_agent, montezuma_alt_agent]
```

Running WorldCoder
```
uv run python make_observations.py task=Pong # choose task from [Pong, PongAlt, MontezumaRevenge, MontezumaRevengeAlt]
uv run python run.py --config-name=pong_agent # choose config-name from [pong_agent, pong_alt_agent, montezuma_agent, montezuma_alt_agent]
```

Running ReAct
```
uv run python run_react.py task=Pong # choose task from [Pong, PongAlt, MontezumaRevenge, MontezumaRevengeAlt]
```

Running PPO
```
uv run python run_rl.py task=Pong # choose task from [Pong, PongAlt, MontezumaRevenge, MontezumaRevengeAlt]
```

## Example learned PoE-World world models

**mr_world_model_seed0.txt** and **pong_world_model_seed0.txt** contain learned PoE-World world models for Montezuma's Revenge and Pong, respectively.

## Important Files

**agents/agent.py**
Contains the implementation of the main agent class, which is responsible for interacting with the environment, calling planning algorithms, and calling functions to update world models.

**agents/mcts.py**
Implements the Monte Carlo Tree Search (MCTS) algorithm, which is used by the agent to plan motions.

**classes/envs**
Contains the environment classes in the style of openai's gym

**classes/helper.py**
Contains various helper classes that are interfaces to game objects (Obj), their interactions (Interaction), game states (ObjList), etc.

**learners/world_model_learner.py**
Implements the world model learner, which calls the obj model learner for all object types.

**learners/obj_model_learner.py**
Implements the object model learner, which calls synthesizers to get programs and calls MoEObjModel in learners/models.py to fit the weights of the programs.

**learners/synthesizer.py**
Contains modules that synthesize programs based on observation

**learners/models.py**
Contains classes that let us fit the weights of the programs

## Citation 

```
@inproceedings{piriyakulkij2025poeworld,
  Author = {Wasu Top Piriyakulkij and Yichao Liang and Hao Tang and Adrian Weller and Marta Kryven and Kevin Ellis},
  Title = {PoE-World: Compositional World Modeling with Products of Programmatic Experts}, 
  Year = {2025},
  booktitle={Advances in Neural Information Processing Systems},
}
```
