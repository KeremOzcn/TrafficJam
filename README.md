# AI Traffic Lights Controller

An AI-powered traffic signal controller that simulates a 4-way intersection and compares multiple intelligent control strategies to minimize vehicle wait times and prevent collisions.

## Methods

| Method | Type | Description |
|--------|------|-------------|
| `fc` | Fixed Cycle | Switches signal every 15 seconds |
| `lqf` | Longest Queue First | Switches to favor the longer queue |
| `logic` | Propositional Logic | Modus Ponens inference rules |
| `math` | Math Model | Poisson arrivals + M/M/1 queuing + Linear Algebra |
| `qlearning` | Reinforcement Learning | Pre-trained Q-Learning agent (10K episodes) |
| `search` | Genetic Algorithm | Evolves optimal action sequences |
| `mcts` | Monte Carlo Tree Search | Lookahead simulation tree search |
| `compare` | Comparison | Runs all methods and produces a comparison report |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py -m [method] -e [episodes] [-r]
```

- `-m` / `--method` : control method (see table above)
- `-e` / `--episodes` : number of evaluation episodes
- `-r` / `--render` : show the simulation window (optional)

## Examples

```bash
# Run Q-Learning for 10 episodes with visualization
python main.py -m qlearning -e 10 -r

# Run Genetic Algorithm for 5 episodes
python main.py -m search -e 5

# Compare all methods over 10 episodes and save charts
python main.py -m compare -e 10

# Run Propositional Logic agent
python main.py -m logic -e 10 -r
```

## Window Controls (when using -r)

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `+` | Speed up simulation |
| `-` | Slow down simulation |
| Mouse wheel | Zoom in/out |
| Left drag | Pan view |

## Project Structure

```
AI-Traffic-Lights-Controller/
├── main.py
├── DefaultCycles/          # FC and LQF methods
├── Logic/                  # Propositional Logic + Modus Ponens
├── MathModel/              # Probability (Poisson/M/M/1) + Linear Algebra
├── ReinforcementLearning/  # Q-Learning agent
├── Search/                 # Genetic Algorithm + MCTS
└── TrafficSimulator/       # Simulation engine, physics, visualization
```
