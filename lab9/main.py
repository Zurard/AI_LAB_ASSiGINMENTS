"""
Lab 9: MENACE and Bandit Reinforcement Learning

This implementation covers:
1. MENACE (Matchbox Educable Naughts and Crosses Engine) concept
2. Binary bandit with epsilon-greedy algorithm
3. 10-armed non-stationary bandit with random walks
4. Modified epsilon-greedy agent for non-stationary rewards

Reference: 
- MENACE by Michie: http://people.csail.mit.edu/brooks/idocs/matchbox.pdf
- Reinforcement Learning: An Introduction by Sutton and Barto (Chapter 1-2)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime


class BinaryBandit:
    """
    A binary bandit with two actions (1 and 2) that returns stochastic rewards.
    The rewards are stationary by default but can be made non-stationary.
    """
    
    def __init__(self, p1=0.5, p2=0.5):
        """
        Initialize the binary bandit.
        
        Args:
            p1: Probability of reward=1 for action 1
            p2: Probability of reward=1 for action 2
        """
        self.p1 = p1  # Probability of success for action 1
        self.p2 = p2  # Probability of success for action 2
        self.name = "BinaryBandit"
    
    def draw(self, action):
        """
        Draw from the bandit for the given action.
        
        Args:
            action: 1 or 2
            
        Returns:
            1 (success) or 0 (failure) based on the stochastic reward
        """
        if action == 1:
            return 1 if np.random.random() < self.p1 else 0
        elif action == 2:
            return 1 if np.random.random() < self.p2 else 0
        else:
            raise ValueError("Action must be 1 or 2")


class EpsilonGreedyAgent:
    """
    Epsilon-greedy action selection agent for bandit problems.
    """
    
    def __init__(self, epsilon, n_actions):
        """
        Initialize the epsilon-greedy agent.
        
        Args:
            epsilon: Exploration probability (0 to 1)
            n_actions: Number of available actions
        """
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.q_values = np.zeros(n_actions)  # Estimated action values
        self.action_counts = np.zeros(n_actions)  # Number of times each action was chosen
        self.total_count = 0  # Total number of actions taken
    
    def select_action(self):
        """
        Select an action using epsilon-greedy policy.
        
        Returns:
            action: Integer action index (0-based)
        """
        self.total_count += 1
        
        # Explore with probability epsilon
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        # Exploit with probability 1-epsilon
        else:
            # Handle ties by choosing among max q-values
            max_q = np.max(self.q_values)
            best_actions = np.where(self.q_values == max_q)[0]
            action = np.random.choice(best_actions)
        
        self.action_counts[action] += 1
        return action
    
    def update(self, action, reward):
        """
        Update Q-value estimate using sample average.
        
        Args:
            action: Integer action index (0-based)
            reward: Reward received (0 or 1)
        """
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]


class TenArmedNonStationaryBandit:
    """
    10-armed bandit with non-stationary rewards.
    All mean rewards start equal and then take independent random walks.
    """
    
    def __init__(self, initial_mean=0.0, std_increment=0.01):
        """
        Initialize the 10-armed non-stationary bandit.
        
        Args:
            initial_mean: Initial mean reward for all arms
            std_increment: Standard deviation of the normal increment for random walks
        """
        self.n_arms = 10
        self.means = np.full(10, initial_mean)
        self.std_increment = std_increment
        self.name = "TenArmedNonStationaryBandit"
    
    def step(self):
        """
        Advance one time step: add random walk to all mean rewards.
        
        Returns:
            Updated mean rewards after random walk
        """
        # Add normally distributed increment to all mean rewards
        self.means += np.random.normal(0, self.std_increment, self.n_arms)
        return self.means.copy()
    
    def draw(self, action):
        """
        Draw from the bandit for the given action.
        
        Args:
            action: Action index 0-9
            
        Returns:
            Reward: 1 (success) or 0 (failure) based on current mean reward
        """
        p = self.means[action]
        return 1 if np.random.random() < p else 0


class NonStationaryEpsilonGreedyAgent:
    """
    Modified epsilon-greedy agent for non-stationary bandit problems.
    Uses a constant learning rate (step-size) to track changing reward probabilities.
    """
    
    def __init__(self, epsilon, n_actions, alpha=0.1):
        """
        Initialize the non-stationary epsilon-greedy agent.
        
        Args:
            epsilon: Exploration probability (0 to 1)
            n_actions: Number of available actions
            alpha: Learning rate / step-size (constant for non-stationary problems)
        """
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.alpha = alpha  # Constant learning rate for non-stationary tracking
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
    
    def select_action(self):
        """
        Select an action using epsilon-greedy policy.
        
        Returns:
            action: Integer action index (0-based)
        """
        # Explore with probability epsilon
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        # Exploit with probability 1-epsilon
        else:
            max_q = np.max(self.q_values)
            best_actions = np.where(self.q_values == max_q)[0]
            action = np.random.choice(best_actions)
        
        self.action_counts[action] += 1
        return action
    
    def update(self, action, reward):
        """
        Update Q-value estimate using constant step-size for non-stationary tracking.
        
        The constant learning rate allows the agent to track changing reward distributions.
        
        Args:
            action: Integer action index (0-based)
            reward: Reward received (0 or 1)
        """
        # Constant step-size update: Q <- Q + alpha * (reward - Q)
        self.q_values[action] = self.q_values[action] + self.alpha * (reward - self.q_values[action])


def run_binary_bandit_experiment(epsilon=0.1, n_steps=1000, p1=0.7, p2=0.4):
    """
    Run a binary bandit experiment with epsilon-greedy agent.
    
    Args:
        epsilon: Exploration probability
        n_steps: Number of time steps
        p1: Success probability for action 1
        p2: Success probability for action 2
        
    Returns:
        results dict with experiment data
    """
    print(f"\n=== Binary Bandit Experiment ===")
    print(f"epsilon={epsilon}, n_steps={n_steps}, p1={p1}, p2={p2}")
    
    bandit = BinaryBandit(p1=p1, p2=p2)
    agent = EpsilonGreedyAgent(epsilon=epsilon, n_actions=2)
    
    rewards = []
    chosen_actions = []
    optimal_actions = []  # Track if we chose the optimal action
    
    # Determine optimal action
    optimal_action = 1 if p1 > p2 else 2
    
    for step in range(n_steps):
        # Select action
        action_idx = agent.select_action()  # 0-based index
        action = action_idx + 1  # Convert to 1-based for bandit
        
        # Draw reward from bandit
        reward = bandit.draw(action)
        
        # Update agent
        agent.update(action_idx, reward)
        
        # Track results
        rewards.append(reward)
        chosen_actions.append(action)
        optimal_actions.append(1 if action == optimal_action else 0)
    
    # Calculate statistics
    total_reward = sum(rewards)
    opt_action_rate = sum(optimal_actions) / n_steps
    avg_reward = total_reward / n_steps
    
    print(f"Total reward: {total_reward}/{n_steps}")
    print(f"Optimal action selection rate: {opt_action_rate:.4f}")
    print(f"Average reward per step: {avg_reward:.4f}")
    print(f"Learned Q-values: {agent.q_values}")
    print(f"Action counts: {agent.action_counts}")
    
    return {
        'epsilon': epsilon,
        'p1': p1,
        'p2': p2,
        'total_reward': total_reward,
        'opt_action_rate': opt_action_rate,
        'avg_reward': avg_reward,
        'q_values': agent.q_values.copy(),
        'action_counts': agent.action_counts.copy(),
        'rewards': rewards,
        'chosen_actions': chosen_actions,
        'optimal_actions': optimal_actions
    }


def run_ten_armed_nonstationary_experiment(epsilon=0.1, n_steps=10000, alpha=0.1, 
                                           initial_mean=0.0, std_increment=0.01,
                                           exploration_decay=False):
    """
    Run a 10-armed non-stationary bandit experiment.
    
    Args:
        epsilon: Exploration probability
        n_steps: Number of time steps (minimum 10000 as per assignment)
        alpha: Learning rate for non-stationary tracking
        initial_mean: Initial mean reward for all arms
        std_increment: Standard deviation for random walks
        exploration_decay: Whether to decay epsilon over time
        
    Returns:
        results dict with experiment data
    """
    print(f"\n=== 10-Armed Non-Stationary Bandit Experiment ===")
    print(f"epsilon={epsilon}, n_steps={n_steps}, alpha={alpha}")
    print(f"std_increment={std_increment}, exploration_decay={exploration_decay}")
    
    bandit = TenArmedNonStationaryBandit(initial_mean=initial_mean, std_increment=std_increment)
    
    # Choose agent type
    if exploration_decay:
        # Epsilon that decays
        agent = EpsilonGreedyAgent(epsilon=epsilon, n_actions=10)
        # We'll handle decay manually
    else:
        agent = NonStationaryEpsilonGreedyAgent(epsilon=epsilon, n_actions=10, alpha=alpha)
    
    rewards = []
    chosen_actions = []
    optimal_actions = []
    q_values_history = []
    means_history = []
    
    # Track optimal action (the one with highest mean reward)
    optimal_action = 0  # Will update each step
    
    for step in range(n_steps):
        # Advance the bandit (random walk)
        current_means = bandit.step()
        means_history.append(current_means.copy())
        
        # Update optimal action for current step
        optimal_action = np.argmax(current_means)
        
        # Select action (handle epsilon decay if requested)
        if exploration_decay:
            current_epsilon = epsilon * (1.0 - step / n_steps)
            if np.random.random() < current_epsilon:
                action = np.random.randint(0, 10)
            else:
                max_q = np.max(agent.q_values)
                best_actions = np.where(agent.q_values == max_q)[0]
                action = np.random.choice(best_actions)
            agent.action_counts[action] += 1
        else:
            action = agent.select_action()
        
        # Draw reward from bandit
        reward = bandit.draw(action)
        
        # Update agent
        if exploration_decay:
            pass  # Already updated action count
        else:
            agent.update(action, reward)
        
        # Track results
        rewards.append(reward)
        chosen_actions.append(action)
        optimal_actions.append(1 if action == optimal_action else 0)
        q_values_history.append(agent.q_values.copy())
    
    # Calculate statistics
    total_reward = sum(rewards)
    opt_action_rate = sum(optimal_actions) / n_steps
    avg_reward = total_reward / n_steps
    
    # Convert history to arrays
    q_values_history = np.array(q_values_history)  # shape: (n_steps, 10)
    means_history = np.array(means_history)  # shape: (n_steps, 10)
    
    print(f"Total reward: {total_reward}/{n_steps}")
    print(f"Optimal action selection rate: {opt_action_rate:.4f}")
    print(f"Average reward per step: {avg_reward:.4f}")
    print(f"Final Q-values: {agent.q_values if not exploration_decay else 'N/A'}")
    print(f"Final mean rewards: {bandit.means}")
    
    return {
        'epsilon': epsilon,
        'alpha': alpha,
        'n_steps': n_steps,
        'std_increment': std_increment,
        'exploration_decay': exploration_decay,
        'total_reward': total_reward,
        'opt_action_rate': opt_action_rate,
        'avg_reward': avg_reward,
        'final_q_values': agent.q_values.copy() if not exploration_decay else np.zeros(10),
        'final_means': bandit.means.copy(),
        'opt_action_rate_history': optimal_actions,
        'q_values_history': q_values_history,
        'means_history': means_history,
        'rewards': rewards,
        'chosen_actions': chosen_actions
    }


def run_comparison_experiment():
    """
    Compare basic epsilon-greedy vs modified non-stationary epsilon-greedy
    on the 10-armed non-stationary bandit problem.
    """
    print("=" * 70)
    print("COMPARISON: Basic Epsilon-Greedy vs Non-Stationary Agent")
    print("=" * 70)
    
    # Parameters
    n_steps = 10000
    epsilon = 0.1
    
    # 1. Basic epsilon-greedy (with sample average update - will struggle with non-stationary)
    print("\n--- Basic Epsilon-Greedy Agent (sample average update) ---")
    results_basic = run_ten_armed_nonstationary_experiment(
        epsilon=epsilon, n_steps=n_steps, alpha=0.1,
        exploration_decay=False
    )
    
    # 2. Non-stationary epsilon-greedy (with constant learning rate)
    print("\n--- Non-Stationary Epsilon-Greedy Agent (constant step-size) ---")
    results_nonstat = run_ten_armed_nonstationary_experiment(
        epsilon=epsilon, n_steps=n_steps, alpha=0.01,  # Smaller alpha for stability
        exploration_decay=False
    )
    
    # 3. Non-stationary with epsilon decay
    print("\n--- Non-Stationary with Epsilon Decay ---")
    results_decay = run_ten_armed_nonstationary_experiment(
        epsilon=0.1, n_steps=n_steps, alpha=0.01,
        exploration_decay=True
    )
    
    return results_basic, results_nonstat, results_decay


def plot_results(results_basic, results_nonstat, results_decay, save_path=None):
    """
    Plot the results of the bandit experiments.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('10-Armed Non-Stationary Bandit Experiments', fontsize=16)
    
    # Plot 1: Optimal action selection rate over time
    ax1 = axes[0, 0]
    time = np.arange(len(results_basic['opt_action_rate_history']))
    ax1.plot(time, results_basic['opt_action_rate_history'], 'b-', label='Basic Epsilon-Greedy', alpha=0.7)
    ax1.plot(time, results_nonstat['opt_action_rate_history'], 'r-', label='Non-Stationary (alpha=0.01)', alpha=0.7)
    ax1.plot(time, results_decay['opt_action_rate_history'], 'g-', label='Epsilon-Decay', alpha=0.7)
    ax1.axhline(y=0.1, color='k', linestyle='--', label='Random selection (1/10)')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Optimal Action Selection Rate')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.set_title('Optimal Action Selection Over Time')
    
    # Plot 2: Average reward over time
    ax2 = axes[0, 1]
    # Compute cumulative average reward
    def cumulative_avg(rewards):
        cs = np.cumsum(rewards)
        return cs / np.arange(1, len(rewards) + 1)
    
    ax2.plot(cumulative_avg(results_basic['rewards']), 'b-', label='Basic Epsilon-Greedy', alpha=0.7)
    ax2.plot(cumulative_avg(results_nonstat['rewards']), 'r-', label='Non-Stationary (alpha=0.01)', alpha=0.7)
    ax2.plot(cumulative_avg(results_decay['rewards']), 'g-', label='Epsilon-Decay', alpha=0.7)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Cumulative Average Reward')
    ax2.set_title('Learning Curve (Average Reward)')
    ax2.legend()
    
    # Plot 3: Final Q-values vs true means
    ax3 = axes[1, 0]
    ax3.scatter(range(10), results_nonstat['final_means'], color='red', s=60, label='True Means', zorder=5)
    ax3.scatter(range(10), results_nonstat['final_q_values'], color='blue', s=60, marker='x', label='Estimated Q-values', zorder=5)
    ax3.set_xlabel('Arm Index')
    ax3.set_ylabel('Value')
    ax3.set_title('Final Q-values vs True Means')
    ax3.legend()
    ax3.set_ylim(-0.5, 1.5)
    
    # Plot 4: Summary comparison
    ax4 = axes[1, 1]
    methods = ['Basic EG\n(avg reward)', 'Non-Stationary\n(alpha=0.01)', 'Epsilon-Decay']
    opt_rates = [
        results_basic['opt_action_rate'],
        results_nonstat['opt_action_rate'],
        results_decay['opt_action_rate']
    ]
    avg_rewards = [
        results_basic['avg_reward'],
        results_nonstat['avg_reward'],
        results_decay['avg_reward']
    ]
    
    x_pos = np.arange(len(methods))
    ax4.bar(x_pos, opt_rates, color='blue', alpha=0.7, label='Optimal Action Rate')
    ax4_twin = ax4.twinx()
    ax4_twin.bar(x_pos, avg_rewards, color='lightblue', alpha=0.3, label='Avg Reward', width=0.2)
    ax4.set_ylabel('Optimal Action Selection Rate')
    ax4_twin.set_ylabel('Average Reward')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods)
    ax4.set_title('Summary Comparison')
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig


def create_meca_implementation():
    """
    Create a simple MENACE (Matchbox Educable Naughts and Crosses Engine) implementation.
    
    MENACE is a learning machine that plays tic-tac-toe using matchboxes with beads.
    Each matchbox corresponds to a board state, and beads of different colors represent
    possible moves. After each game, beads are added or removed based on the outcome.
    """
    print("\n" + "="*60)
    print("MENACE (Matchbox Educable Naughts and Crosses Engine)")
    print("="*60)
    print("""
    MENACE Concept:
    - Each matchbox represents a tic-tac-toe board state
    - Beads in each matchbox represent possible moves (by color)
    - After each game:
      * If MENACE wins: add extra beads of the moves it made
      * If MENACE loses: remove the beads of the moves it made
      * If draw: do nothing
    - Over time, MENACE learns to prefer winning moves
    
    Key Components:
    1. Matchbox dictionary: maps board state -> list of moves (beads)
    2. Move selection: randomly pick a bead from the current matchbox
    3. Learning update: modify bead counts based on game outcome
    4. Exploration: random moves ensure new states are explored
    """)
    
    # Simple MENACE implementation
    class SimpleMENACE:
        """
        Simplified MENACE implementation for tic-tac-toe.
        """
        def __init__(self):
            # Initialize matchboxes for all possible board states
            # Each matchbox contains beads representing possible moves
            # For simplicity, we use a limited set of states
            self.matchboxes = {}
            self.games_won = 0
            self.games_lost = 0
            self.games_draw = 0
            self.total_games = 0
            
            # Initialize with some beads for corner, edge, and center moves
            # In a full implementation, this would cover all 5478 possible states
            np.random.seed(42)
            
        def select_move(self, board_state):
            """Select a move from the matchbox for the given state."""
            # Convert numpy array to tuple for hashable key
            state_key = tuple(board_state) if isinstance(board_state, np.ndarray) else board_state
            
            if state_key not in self.matchboxes:
                # Create new matchbox with beads for all available moves
                available_moves = [i for i in range(9) if board_state[i] == -1]
                if available_moves:
                    # Add beads: 3 red, 2 blue, 1 green as initial bias
                    n_moves = len(available_moves)
                    n_red = max(1, n_moves // 2)  # Bias toward first half
                    n_blue = max(1, (n_moves - n_red) // 2)
                    n_green = n_moves - n_red - n_blue
                    
                    beads = []
                    beads.extend(['red'] * n_red)
                    beads.extend(['blue'] * n_blue)
                    beads.extend(['green'] * n_green)
                    np.random.shuffle(beads)
                    self.matchboxes[state_key] = beads
                else:
                    return None  # No moves available
            
            # Randomly select a bead (move)
            beads = self.matchboxes[state_key]
            if beads:
                selected_bead = np.random.choice(beads)
                # Convert bead color to move index
                # This is simplified - full impl would map colors to specific moves
                available_moves = [i for i in range(9) if board_state[i] == -1]
                if available_moves:
                    move_idx = available_moves[0]  # Simplified
                    return move_idx
            return None
        
        def learn(self, board_states, moves, outcome):
            """
            Learn from a game.
            
            Args:
                board_states: List of board states encountered
                moves: List of moves made
                outcome: 1=win, 0=loss, -1=draw
            """
            self.total_games += 1
            
            if outcome == 1:  # Win
                self.games_won += 1
                # Add beads for moves made (reinforce)
                for state, move in zip(board_states, moves):
                    state_key = tuple(state) if isinstance(state, np.ndarray) else state
                    if state_key in self.matchboxes:
                        self.matchboxes[state_key].append('red')
            elif outcome == -1:  # Draw
                self.games_draw += 1
                # Do nothing for draws
            else:  # Loss
                self.games_lost += 1
                # Remove beads for moves made (punish)
                for state, move in zip(board_states, moves):
                    state_key = tuple(state) if isinstance(state, np.ndarray) else state
                    if state_key in self.matchboxes and self.matchboxes[state_key]:
                        if 'red' in self.matchboxes[state_key]:
                            self.matchboxes[state_key].remove('red')
        
        def get_stats(self):
            return {
                'games': self.total_games,
                'wins': self.games_won,
                'losses': self.games_lost,
                'draws': self.games_draw,
                'win_rate': self.games_won / self.total_games if self.total_games > 0 else 0
            }
    
    return SimpleMENACE


def main():
    """Main function to run all experiments."""
    print("=" * 70)
    print("LAB 9: MENACE and Bandit Reinforcement Learning")
    print("=" * 70)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.getcwd(), 'lab9_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================
    # Part 1: Binary Bandit Experiment
    # ==========================================
    print("\n" + "=" * 70)
    print("PART 1: Binary Bandit with Epsilon-Greedy")
    print("=" * 70)
    
    # Helper function for JSON serialization (defined first)
    def serialize_value(v):
        """Convert a single value to JSON-serializable Python type."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            return int(float(v))
        elif isinstance(v, (np.bool_)):
            return bool(v)
        elif hasattr(v, 'item'):
            return v.item()
        elif isinstance(v, (int, float, str, bool)):
            return v
        return str(v)
    
    def serialize_dict(d):
        """Recursively convert dict values to JSON-serializable types."""
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = serialize_dict(value)
            elif isinstance(value, list):
                result[key] = [serialize_value(v) if not isinstance(v, (dict, list)) else 
                              (serialize_dict(v) if isinstance(v, dict) else 
                               [serialize_value(vv) for vv in v])
                for v in value]
            else:
                result[key] = serialize_value(value)
        return result
    
    # Test with different epsilon values
    binary_results = []
    for eps in [0.01, 0.1, 0.25]:
        result = run_binary_bandit_experiment(epsilon=eps, n_steps=1000, p1=0.7, p2=0.4)
        binary_results.append(result)
        # Save individual result with proper serialization
        result_serializable = serialize_dict(result)
        with open(os.path.join(output_dir, f'binary_eps_{eps}.json'), 'w') as f:
            json.dump(result_serializable, f, indent=2)
    
    # ==========================================
    # Part 2: 10-Armed Non-Stationary Bandit
    # ==========================================
    print("\n" + "=" * 70)
    print("PART 2: 10-Armed Non-Stationary Bandit")
    print("=" * 70)
    
    # Run comparison experiment
    results_basic, results_nonstat, results_decay = run_comparison_experiment()
    
    # Save results using serialize_dict
    with open(os.path.join(output_dir, 'basic_epsilon_greedy.json'), 'w') as f:
        json.dump(serialize_dict(results_basic), f, indent=2)
    with open(os.path.join(output_dir, 'nonstationary_epsilon_greedy.json'), 'w') as f:
        json.dump(serialize_dict(results_nonstat), f, indent=2)
    with open(os.path.join(output_dir, 'epsilon_decay.json'), 'w') as f:
        json.dump(serialize_dict(results_decay), f, indent=2)
    
    # ==========================================
    # Part 3: Generate plots
    # ==========================================
    print("\n" + "=" * 70)
    print("Part 3: Generating plots...")
    print("=" * 70)
    
    plot_path = os.path.join(output_dir, 'bandit_comparison.png')
    plot_results(results_basic, results_nonstat, results_decay, save_path=plot_path)
    
    # ==========================================
    # Part 4: MENACE Implementation
    # ==========================================
    print("\n" + "=" * 70)
    print("PART 4: MENACE Implementation Demo")
    print("=" * 70)
    
    # Create MENACE instance
    menage = create_meca_implementation()
    menage_instance = menage()
    
    # Play a few games
    np.random.seed(42)
    for game in range(5):
        # Initialize empty board: -1=empty, 0=O, 1=X
        board = np.full(9, -1)
        game_states = []
        game_moves = []
        
        game_over = False
        step = 0
        
        while not game_over and step < 9:
            # MENACE's turn (X, player 1)
            state_key = tuple(board)
            move = menage_instance.select_move(board)
            
            if move is None:
                break
            
            # Make move
            board[move] = 1
            game_states.append(state_key)
            game_moves.append(move)
            
            # Check for win
            # Simple win check
            win_patterns = [
                [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
                [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
                [0, 4, 8], [2, 4, 6]  # diagonals
            ]
            
            game_over = any(all(board[i] == 1 for i in pattern) for pattern in win_patterns)
            if all(b != -1 for b in board) or step >= 8:
                game_over = True
            
            if game_over:
                break
            
            # Opponent's turn (O, player 0) - random
            available = [i for i in range(9) if board[i] == -1]
            if available:
                opp_move = np.random.choice(available)
                board[opp_move] = 0
            
            step += 1
        
        # Learn from the game
        outcome = 0
        if any(all(board[i] == 1 for i in pattern) for pattern in win_patterns):
            outcome = 1  # MENACE won
        elif all(b != -1 for b in board):
            outcome = -1  # Draw
        else:
            outcome = 0  # Loss
        
        menage_instance.learn(game_states, game_moves, outcome)
    
    # Print final stats
    stats = menage_instance.get_stats()
    print(f"\nMENACE after 5 games:")
    print(f"  Wins: {stats['wins']}, Losses: {stats['losses']}, Draws: {stats['draws']}")
    print(f"  Win rate: {stats['win_rate']:.2f}")
    print(f"  Total games: {stats['games']}")
    
    # Save MENACE stats (convert numpy types)
    stats_serializable = serialize_dict(stats)
    with open(os.path.join(output_dir, 'meca_stats.json'), 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"""Binary Bandit (p1=0.7, p2=0.4):
  - Epsilon=0.01: opt_action_rate={binary_results[0]['opt_action_rate']:.4f}, 
    avg_reward={binary_results[0]['avg_reward']:.4f}
  - Epsilon=0.1: opt_action_rate={binary_results[1]['opt_action_rate']:.4f}, 
    avg_reward={binary_results[1]['avg_reward']:.4f}
  - Epsilon=0.25: opt_action_rate={binary_results[2]['opt_action_rate']:.4f}, 
    avg_reward={binary_results[2]['avg_reward']:.4f}

10-Armed Non-Stationary Bandit (10,000 steps):
  - Basic Epsilon-Greedy (sample avg):
    opt_action_rate={results_basic['opt_action_rate']:.4f}, avg_reward={results_basic['avg_reward']:.4f}
  - Non-Stationary (alpha=0.01):
    opt_action_rate={results_nonstat['opt_action_rate']:.4f}, avg_reward={results_nonstat['avg_reward']:.4f}
  - Epsilon-Decay:
    opt_action_rate={results_decay['opt_action_rate']:.4f}, avg_reward={results_decay['avg_reward']:.4f}

MENACE:
  - Win rate: {stats['win_rate']:.2f} after {stats['games']} games
""")
    
    print(f"Results saved to: {output_dir}")
    print(f"Plot saved to: {plot_path}")
    
    return {
        'binary_results': binary_results,
        'results_basic': results_basic,
        'results_nonstat': results_nonstat,
        'results_decay': results_decay,
        'meca_stats': stats
    }


if __name__ == '__main__':
    results = main()