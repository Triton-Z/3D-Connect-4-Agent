# 3D Connect 4 AI Agent using AlphaZero

This project implements an AI that learns to play 3D Connect 4 using the AlphaZero algorithm, which combines Monte Carlo Tree Search (MCTS) and a deep convolutional + residual neural network trained through self-play.

> [!WARNING]
> The model should be trained for ~10,000 iterations. However, this has not been completed due to hardware constraints. Despite this, the full architecture is implemented and ready for training and experimentation.


## Overview

The game is a 3D version of Connect 4 played on a 7×7×6 grid. Players drop pieces into columns, and the first to connect 4 in any direction (including diagonals through 3D space) wins.

The AI itself is modeled after DeepMind’s AlphaZero (read more about it [here](https://arxiv.org/abs/1712.01815)). In short, its strategy is to combine a Monte Carlo Search Tree with a deep residual neural network.

The AI learns through self-play and updates the network based on game outcomes and MCTS-guided policy distributions.

> [!NOTE]
> This implementation generalizes the 2D AlphaZero approach to a full 3D environment, significantly increasing complexity and the search space.


## Features

- Fully configurable game parameters (board size, win length)
- Deep 3D CNN with residual blocks
- AlphaZero-style self-play training pipeline
- Dirichlet noise for exploration
- Full MCTS implementation with PUCT scoring
- Modular architecture, easy to extend or adapt

## How It Works

1. **Game Engine**  
   Encodes game rules, valid moves, next-state transitions, and win conditions.

2. **Neural Network**  
   - **Input**: A 3-channel tensor encoding for player pieces, opponent pieces, and player turn.
   - **Output**:
     - A **policy** vector over all legal moves.
     - A **value** prediction estimating the expected outcome.

3. **Monte Carlo Tree Search (MCTS)**  
   - Uses neural network predictions to guide search.
   - Prioritizes high-value and high-probability moves.
   - Produces improved move probabilities for training.

4. **Training Pipeline (Coach)**  
   - Runs self-play episodes.
   - Stores game data (state, policy, outcome) into a memory buffer.
   - Trains the network using minibatch stochastic gradient descent.

> [!TIP]
> The training loop also includes a temperature schedule to control move exploration in early and late game stages.

## Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow numpy
```

### 2. Run notebook cells

### 3. (Optionally) Load Pretrained Weights

```python
network.model.load_weights("modelIteration_X.weights.h5")
```

> [!IMPORTANT]
> Ensure you save weights after each iteration (either locally or to the cloud) using:
> 
> ```python
> network.model.save_weights("modelIteration_X.weights.h5")
> ```


## Neural Network Architecture

- **Input**: ```(6, 7, 7, 3)``` tensor (height × width × length × channels)
- **Architecture**:
  - Initial Conv3D layer
  - 4 Residual blocks (Conv3D → BatchNorm → ReLU)
- **Heads**:
  - **Policy Head**: Outputs 49 probabilities (7×7) using softmax
  - **Value Head**: Outputs a scalar between -1 and 1 using tanh


## Configuration (Hyperparameters)

```python
args = {
    'numIterations': 10000,
    'numSimulations': 500,
    'epochs': 10,
    'batchSize': 64,
    'maxBufferSize': 50000,
    'cPuct': 1.25,
    'temperatureMoves': 12,
    'dirichletAlpha': 0.3,
    'dirichletEpsilon': 0.25,
}
```

> [!NOTE]
> You can reduce values (e.g., number of simulations) for faster but less accurate testing or debugging.


## Limitations / Comments

- Full training (~10k iterations) requires significant compute (preferably a GPU).
- Some ```print()``` statements are included in the loop to aid debugging and monitoring; these can be removed or commented out.
- Currently runs only through the CLI or notebook; no GUI or visualizer is yet provided.

> [!CAUTION]
> MCTS with 500 simulations per move can be computationally expensive. Reducing ```numSimulations``` may help if you're on limited hardware.


## Future Improvements

- Add pretrained weights for demo and benchmarking
- Add evaluation matches between different versions
- Integrate a simple GUI or 3D visualization to play against the AI
- Implement performance profiling and optimization (especially for MCTS)
- Consider curriculum learning or reinforcement fine-tuning

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
