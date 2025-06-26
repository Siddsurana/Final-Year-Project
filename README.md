# 5G Energy Optimization using Q-Learning

A reinforcement learning approach to optimize energy consumption in 5G networks by intelligently switching between normal and power-saving modes based on network load conditions.


## Overview

This project implements a Q-Learning based reinforcement learning agent to optimize energy consumption in 5G networks. The system learns to make intelligent decisions about when to switch between normal operation mode and power-saving mode based on real-time network load conditions.

The project addresses the critical challenge of energy efficiency in 5G networks, which is essential for:
- Reducing operational costs for telecom operators
- Minimizing environmental impact
- Extending battery life in mobile devices
- Improving overall network sustainability

## Features

- 🔋 **Intelligent Energy Management**: Automatically switches between normal and power-saving modes
- 📊 **Real-time Load Analysis**: Processes network load data to make optimal decisions
- 🤖 **Q-Learning Implementation**: Uses reinforcement learning for adaptive optimization
- 📈 **Performance Visualization**: Comprehensive plots showing energy consumption patterns
- 📋 **Detailed Analytics**: Summary statistics and performance metrics
- ⚡ **Efficient Algorithm**: Fast convergence with configurable hyperparameters

## Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or Google Colab

### Required Libraries
```bash
pip install numpy matplotlib random
```

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/5g-energy-optimization.git
cd 5g-energy-optimization
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the notebook:**
```bash
jupyter notebook 5g_energy_optimization.ipynb
```

**Or run directly in Google Colab:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RXX7JmzVUvHHS0ycPSSWc6rIlFiUkRoG?usp=sharing)

## Usage

### Quick Start
```python
# Initialize the environment
env = EnergyOptimizationEnv()

# Train the Q-learning agent
q_table = q_learning(env, episodes=1000)

# Test the learned policy
state = env.reset()
done = False
while not done:
    state_idx = state[0]
    action = np.argmax(q_table[state_idx])
    state, reward, done = env.step(action)

# View results
print(f"Total Energy Consumed: {env.total_energy_consumption}")
```

### Customization
You can modify the following parameters:

```python
# Q-learning hyperparameters
episodes = 1000        # Number of training episodes
alpha = 0.1           # Learning rate
gamma = 0.95          # Discount factor
epsilon = 1.0         # Initial exploration rate
epsilon_decay = 0.995 # Exploration decay rate

# Energy consumption values
energy_consumed_normal = 10      # Energy used in normal mode
energy_consumed_power_save = 3   # Energy used in power-saving mode
```

## Algorithm Details

### Environment
The `EnergyOptimizationEnv` class simulates a 5G network environment with:
- **State Space**: Network load values (0-100)
- **Action Space**: 
  - 0 = Normal Mode
  - 1 = Power-Saving Mode
- **Reward System**:
  - +10 for using power-saving mode when load < 20
  - -1 for using power-saving mode when load > 80
  - 0 otherwise

### Q-Learning Implementation
The agent uses Q-learning with:
- **Q-table**: 101 × 2 matrix (load states × actions)
- **Epsilon-greedy exploration**: Balances exploration vs exploitation
- **Bellman equation**: Updates Q-values based on immediate and future rewards

### Decision Logic
- **Low Load (< 20)**: Power-saving mode recommended (lower energy, good performance)
- **High Load (> 80)**: Normal mode required (higher energy, necessary for performance)
- **Medium Load (20-80)**: Flexible based on learned policy

## Results

### Key Findings
- **Energy Savings**: Achieved significant reduction in total energy consumption
- **Smart Switching**: Agent learned to use power-saving mode during low network load periods
- **Performance Balance**: Maintained network performance while optimizing energy usage

### Visualizations
The project generates two main plots:
1. **Energy Consumption per Step**: Shows instantaneous energy usage over time
2. **Cumulative Energy Consumption**: Displays total energy consumption trend

### Sample Performance Metrics
- **Total Energy Consumption**: Varies based on network conditions
- **Power-Saving Mode Usage**: Typically 15-30% depending on load patterns
- **Average Energy per Step**: Reduced compared to always-normal operation
- **Reward Accumulation**: Positive trend indicating successful optimization

## Technologies Used

### Programming Language
- **Python 3.x**

### Core Libraries
- **NumPy**: Numerical computing and array operations
- **Matplotlib**: Data visualization and plotting
- **Random**: Random number generation for exploration

### Machine Learning Approach
- **Q-Learning**: Reinforcement learning algorithm
- **Epsilon-Greedy Strategy**: Exploration-exploitation balance
- **Temporal Difference Learning**: Value function updates

## Project Structure

```
5g-energy-optimization/
├── 5g_energy_optimization.ipynb    # Main Jupyter notebook
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── data/
│   └── network_load_simulation.py  # Network load data generation
├── src/
│   ├── environment.py              # EnergyOptimizationEnv class
│   ├── q_learning.py               # Q-learning algorithm
│   └── visualization.py            # Plotting functions
├── results/
│   ├── energy_consumption_plots.png
│   └── performance_metrics.txt
└── docs/
    ├── algorithm_explanation.md
    └── usage_examples.md
```

## Performance Metrics

### Energy Efficiency Metrics
- **Energy Reduction Percentage**: Compared to baseline (always normal mode)
- **Power-Saving Mode Utilization**: Percentage of time in low-power state
- **Load-Adapted Switching**: Frequency of appropriate mode switching

### Learning Performance
- **Convergence Rate**: Episodes required for stable policy
- **Exploration vs Exploitation**: Balance achieved through epsilon decay
- **Reward Optimization**: Cumulative reward improvement over training

### Network Performance
- **Service Quality Maintenance**: Ensuring performance during high load
- **Response Time**: Speed of mode switching decisions
- **Adaptability**: Performance across different load patterns

## Advanced Features

### Hyperparameter Tuning
Experiment with different values:
- Learning rate (alpha): 0.01 - 0.3
- Discount factor (gamma): 0.8 - 0.99
- Exploration decay: 0.99 - 0.999

### Extended Analysis
- Compare performance across different network load patterns
- Analyze seasonal or daily usage patterns
- Test robustness with noise in load measurements

## Future Enhancements

- **Deep Q-Networks (DQN)**: Scale to larger state spaces
- **Multi-Agent Systems**: Coordinate multiple base stations
- **Real-Time Implementation**: Deploy in actual network infrastructure
- **Advanced Reward Functions**: Include latency and throughput metrics
- **Predictive Models**: Anticipate load changes for proactive optimization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for any changes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Authors** - Rahul Prati, Siddharth Surana 

**Project Link**: [https://github.com/yourusername/5g-energy-optimization](https://github.com/yourusername/5g-energy-optimization)

**Google Colab**: [https://colab.research.google.com/drive/1RXX7JmzVUvHHS0ycPSSWc6rIlFiUkRoG](https://colab.research.google.com/drive/1RXX7JmzVUvHHS0ycPSSWc6rIlFiUkRoG)

## Acknowledgments

- Inspiration from 5G energy efficiency research
- Q-learning algorithm implementation based on Sutton & Barto's "Reinforcement Learning: An Introduction"
- Network simulation concepts from telecommunications literature

## Keywords

`5G` `Energy Optimization` `Q-Learning` `Reinforcement Learning` `Machine Learning` `Network Management` `Power Saving` `Telecommunications` `Python` `AI`
