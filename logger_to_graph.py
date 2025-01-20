from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from collections import defaultdict
import seaborn as sns
import numpy as np
import os

"""
This module is designed to process and analyze log files generated during decentralized federated learning simulations. 
It extracts key metrics, identifies malicious behavior, and generates insightful visualizations to evaluate the 
performance and security of the system under various configurations.

### Key Features
1. **Data Extraction**:
   - **Aggregated Scores**: Extracts validation scores aggregated during each simulation round.
   - **Node Stakes**: Tracks stake evolution of nodes across rounds for Proof-of-Stake configurations.
   - **Peer Training Sets**: Maps nodes to the datasets they use during training.
   - **Gradient Loss**: Extracts gradient loss or training history for active nodes in each round.
   - **Accuracy**: Tracks the accuracy of the aggregated model over rounds.

2. **Visualization**:
   - **Line Plots**: Metrics like accuracy and the number of active nodes per round.
   - **Heatmaps**: Node inclusion scores and aggregation behavior during training rounds.
   - **3D Plots**: Evolution of node stakes for attacker and non-attacker nodes.

3. **Malicious Behavior Detection**:
   - Highlights attacker nodes and visualizes their influence during the simulation.
   - Identifies nodes involved in misclassifications or targeted attacks.

4. **Export Capabilities**:
   - Saves generated plots (e.g., accuracy plots, heatmaps) to specified directories for further analysis.

### Main Functions
- `extract_aggregated_nodes_and_scores(logger_path: str) -> tuple`:
  Extracts validation scores, miner winner data, and validator IDs for each round.
  
- `extract_stakes_per_round(logger_path: str) -> dict`:
  Tracks stake changes for each node across simulation rounds.

- `extract_peer_trainsets(logger_path: str) -> dict`:
  Maps each peer to the training datasets it used.

- `extract_gradient_loss_per_round(logger_path: str, gradient: bool = True) -> dict`:
  Extracts either gradient loss or training history for active nodes.

- `agg_acc_from_logger(logger_path: str) -> list`:
  Retrieves the accuracy of the aggregated model over rounds.

- `num_of_nodes_per_round(logger_path: str) -> list`:
  Tracks the number of active nodes in each training round.

- Visualization functions:
  - `plot_data(data: list, title: str, xlabel: str, ylabel: str, subplot: tuple = (1,1,1))`:
    Generates line plots for extracted metrics.
  - `plot_mean_variance_loss(peer_data: dict, num_epochs: int, gradient: bool = True, subplot: tuple = (1,1,1))`:
    Visualizes mean and variance of metrics over epochs.
  - `plot_stakes_3d(stakes_per_round: dict, attacker_nodes: list, fig, position: tuple)`:
    Creates a 3D plot of stake evolution.
  - `plot_aggregated_nodes_heatmap(aggregated_data: dict, winner_data: dict, validators_data: dict, num_rounds: int, num_nodes: int, fig, position: tuple, attacker_nodes: list)`:
    Renders a heatmap to show node inclusion scores across rounds.

### Usage
- Set the `LOGGER_PATH` to the log file you want to analyze.
- Use flags such as `GRADIENTS`, `POS`, and `HEATMAP` to control the type of analysis and visualization.
- Run the module as a script to generate and optionally save visualizations.

### Dependencies
- **Python Libraries**:
  - `matplotlib`, `seaborn`: For plotting and visualization.
  - `numpy`: For numerical computations.
  - `collections`: To manage extracted data.
  - `re`: For parsing log file patterns.
  - `os`: To manage file paths and save outputs.
- **Log File Structure**:
  - The log file must follow the expected format for parsing round data, validation scores, and node activities.

### Example
1. Run the script:
   ```bash
   python logger_to_graph.py
    ```
"""

def extract_aggregated_nodes_and_scores(logger_path: str) -> tuple:
    aggregated_data = defaultdict(list)  # Dictionary to store round -> [(peer_id, score)]
    tmp_data = {}  # Temporary dictionary to store update name -> (round, peer_id)
    winner_data = {}  # Dictionary to store round -> winner_miner_id
    validator_data = {}  # Dictionary to store round -> validator_id
    round_number = 1

    with open(logger_path, 'r') as f:
        lines = f.readlines()

    for line in lines:

        if "Round" in line:
            round = re.search(r"Round: (\d+)", line)
            if round:
                round_number = int(round.group(1))

        # Check for the "Update written" message to get round, peer ID, and update name
        if "Update written directly on disk" in line:
            round_match = re.search(r"Round: (\d+)", line)
            peer_match = re.search(r"\[Peer-(\d+)\]", line)
            update_match = re.search(r"Update name: ([a-f0-9]+)", line)

            if round_match and peer_match and update_match:
                round_number = int(round_match.group(1))
                peer_id = int(peer_match.group(1))
                update_name = update_match.group(1)
                
                # Store this information in tmp_data for later matching with the score
                tmp_data[update_name] = (round_number, peer_id)

        # Check for the "Update is ready" message to get score for each update
        elif "Update is ready for the aggregation" in line:
            update_match = re.search(r"Update name: ([a-f0-9]+)", line)
            score_match = re.search(r"Aggregated validation score: ([\d.]+)", line)

            if update_match and score_match:
                update_name = update_match.group(1)
                score = float(score_match.group(1))
                
                # Retrieve round and peer ID using update_name
                if update_name in tmp_data:
                    round_number, peer_id = tmp_data[update_name]
                    
                    # Append the (peer_id, score) tuple to the appropriate round in aggregated_data
                    aggregated_data[round_number].append((peer_id, score))
        
        elif "Update has enough positive scores" in line:
            update_match = re.search(r"Update name: ([a-f0-9]+)", line)
            score_match = re.search(r"Aggregated score: ([\d.]+)", line)

            if update_match and score_match:
                update_name = update_match.group(1)
                score = float(score_match.group(1))
                
                # Retrieve round and peer ID using update_name
                if update_name in tmp_data:
                    round_number, peer_id = tmp_data[update_name]
                    
                    # Append the (peer_id, score) tuple to the appropriate round in aggregated_data
                    aggregated_data[round_number].append((peer_id, score))
        
        elif "passed the validation" in line:
            update_match = re.search(r"Update ([a-f0-9]+)", line)
            score_match = re.search(r"Aggregated score assigned: ([\d.]+)", line)
            if update_match and score_match:
                update_name = update_match.group(1)
                score = float(score_match.group(1))
                if update_name in tmp_data:
                    round_number, peer_id = tmp_data[update_name]
                    aggregated_data[round_number].append((peer_id, score))

        # Capture winner miner data for each round
        elif "Winner miner of this round elected" in line:
            round_match = re.search(r"Round: (\d+)", line)
            winner_match = re.search(r"Winner miner: \[(\d+)\]", line)

            if round_match and winner_match:
                round_number = int(round_match.group(1))
                winner_id = int(winner_match.group(1))
                winner_data[round_number] = winner_id
        
        elif "Node responsible of model aggregation elected" in line:
            aggrgegator_match = re.search(r"Node id: (\d+)", line)
            if aggrgegator_match:
                aggregator_id = int(aggrgegator_match.group(1))
                winner_data[round_number] = aggregator_id
        
        elif "New validators elected" in line or "First validators" in line:
            validators_match = re.search(r"Validators ids: \[(.*?)\]", line)
            first_validators = re.search(r"First validators: \[(.*?)\]", line)
            committee_val_match = re.search(r"Validator ids: \[(.*?)\]", line)

            if validators_match:
                validators_ids = list(map(int, validators_match.group(1).split(",")))
                validator_data[round_number] = validators_ids
            elif first_validators:
                validators_ids = list(map(int, first_validators.group(1).split(",")))
                validator_data[round_number] = validators_ids
            elif committee_val_match:
                validators_ids = list(map(int, committee_val_match.group(1).split(",")))
                validator_data[round_number] = validators_ids

    return aggregated_data, winner_data, validator_data

def extract_stakes_per_round(logger_path: str) -> dict:
    stakes_per_round = defaultdict(dict)
    current_round = None  # To keep track of the most recent round number
    
    with open(logger_path, 'r') as f:
        lines = f.readlines()
    

    for i, line in enumerate(lines):
        # Check for the line with the round number
        if "Accuracy of the new aggregated model on the test set computed" in line:
            round_match = re.search(r"Round: (\d+)", line)
            if round_match:
                current_round = int(round_match.group(1))

        # Look for the line with "Node stakes"
        if "Node stakes:" in line:
            stakes_match = re.search(r"Node stakes: ({.*)(}|\.\.\.)", line)
            if stakes_match and current_round is not None:
                # Extract the dictionary string and evaluate it to convert to dict
                stakes_str = stakes_match.group(1)
                if "..." in stakes_str:
                    stakes_str = stakes_str.split("...")[0]
                    stakes_str = stakes_str.strip("{")
                else:
                    stakes_str = stakes_str.strip("{}")
                try:
                    # Evaluate the dictionary safely
                    stakes = {int(k): int(v) for item in stakes_str.split(", ") if item and ":" in item for k, v in [item.split(":")]}
                    if current_round is not None:
                        stakes_per_round[current_round] = stakes
                    else:
                        print(f"Stakes found without a corresponding round: {stakes}")
                except ValueError:
                    print(f"Error parsing node stakes: {stakes_str}")     
            else:
                print("No match found or round not set for line:", line)

    return stakes_per_round

def extract_peer_trainsets(logger_path: str) -> dict:
    peer_trainsets = {}
    with open(logger_path, 'r') as f:
        lines = f.readlines()
        
    line_starter = 'DEBUG:'
    trainset_message = "Quanta paths used:"
    
    for line in lines:
        if line.startswith(line_starter) and trainset_message in line:
            peer_id = int(re.search(r"\[Peer-(\d+)\]", line).group(1))
            trainset_ids = re.findall(r"trainset_(\d+).npz", line)
            trainset_ids = list(map(int, trainset_ids))
            peer_trainsets[peer_id] = trainset_ids
    return peer_trainsets

def extract_gradient_loss_per_round(logger_path: str, gradient: bool = True) -> dict:
    peer_data = defaultdict(list)
    active_peers_per_round = defaultdict(set)

    with open(logger_path, 'r') as f:
        lines = f.readlines()

    line_starter = 'INFO:'
    active_message = "Node is an active trainer in the next round"
    
    # First pass to gather active peers per round
    for line in lines:
        if line.startswith(line_starter) and active_message in line:
            round_number = int(line.split("Round:")[1].split('.')[0].strip())
            peer_id = int(re.search(r'\[Peer-(\d+)\]', line).group(1))
            active_peers_per_round[round_number].add(peer_id)
    
    line_starter = 'DEBUG:'
    if gradient:
        message = "Gradient loss computed:"
    else:
        message = "History of the training"
    
    for line in lines:
        if "Round:" in line:
            try:
                round_number = int(re.search(r"Round: (\d+)", line).group(1))
            except AttributeError:
                print(f"Error parsing round number in line: {line}")
                continue
        if line.startswith(line_starter) and message in line:
            peer_id = int(re.search(r"\[Peer-(\d+)\]", line).group(1))
            if gradient:
                gradient_loss = float(line.split("Gradient loss computed:")[1].strip())
                peer_data[peer_id].append((round_number, gradient_loss))
            else:
                accuracy_values = re.search(r"'accuracy': \[(.*?)\]", line).group(1)
                accuracy_list = list(map(float, accuracy_values.split(", ")))
                last_accuracy = accuracy_list[-1]
                peer_data[peer_id].append((round_number, last_accuracy))

    return peer_data

def agg_acc_from_logger(logger_path: str) -> list:
    with open(logger_path, 'r') as f:
        lines = f.readlines()
    data = []
    line_starter = 'INFO:'
    message_contains = 'Accuracy of the new aggregated model on the test set computed'
    for line in lines:
        # Check if line starts with `line_starter` and contains the specified message
        if line.startswith(line_starter) and message_contains in line:
            # Extract the accuracy value after "Accuracy:" keyword
            accuracy_value = float(line.split("Accuracy:")[1].split()[0][:-1])
            data.append(accuracy_value)
    return data

def num_of_nodes_per_round(logger_path: str) -> list:
    with open(logger_path, 'r') as f:
            lines = f.readlines()
        
    round_nodes_count = {}
    line_starter = 'INFO:'
    message_contains = "Node is an active trainer in the next round"
    for line in lines:
        if line.startswith(line_starter) and message_contains in line:
            # Extract the round number from the line
            round_number = int(line.split("Round:")[1].split('.')[0].strip())
            
            # Count nodes per round
            if round_number not in round_nodes_count:
                round_nodes_count[round_number] = 1
            else:
                round_nodes_count[round_number] += 1
    
    # Convert dictionary to a list of node counts, sorted by round number
    max_round = max(round_nodes_count.keys())
    node_counts_per_round = [round_nodes_count.get(round, 0) for round in range(1, max_round + 1)]
    return node_counts_per_round
    
def plot_data(data: list, title: str, xlabel: str, ylabel: str, subplot: tuple = (1,1,1)) -> None:
    ax = plt.subplot(*subplot)
    ax.plot(data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_mean_variance_loss(peer_data: dict, num_epochs: int, gradient: bool = True, subplot:tuple = (1,1,1)) -> None:
    epoch_data = defaultdict(list)
    
    for peer_id, data in peer_data.items():
        for round_number, measure in data:
            epoch_data[round_number].append(measure)
    
    means = []
    variances = []
    for epoch in range(num_epochs):
        data = epoch_data[epoch+1]
        mean = np.mean(data) if data else 0
        variance = np.var(data) if data else 0
        means.append(mean)
        variances.append(variance)
    
    ax = plt.subplot(*subplot)
    epochs = range(1, num_epochs + 1)
    if gradient:
        sns.lineplot(x=epochs, y=means, label="Mean Gradient Loss", color="b", ax=ax)
    else:
        sns.lineplot(x=epochs, y=means, label="Mean Accuracy", color="b", ax=ax)
    
    ax.fill_between(epochs, 
                    np.array(means) - np.sqrt(variances), 
                    np.array(means) + np.sqrt(variances), 
                    color="b", alpha=0.3, label="Variance")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Loss' if gradient else 'Accuracy')
    ax.legend()
    ax.set_title('Mean Gradient Loss with Variance Over Epochs' if gradient else 'Mean Accuracy with Variance Over Epochs')

def plot_stakes_3d(stakes_per_round: dict, attacker_nodes: list, fig, position: tuple) -> None:
    ax = fig.add_subplot(*position, projection='3d')

    rounds_attackers = []
    node_ids_attackers = []
    stakes_attackers = []
    rounds_non_attackers = []
    node_ids_non_attackers = []
    stakes_non_attackers = []

    # Organize data for 3D plotting
    for round_number, stake_dict in stakes_per_round.items():
        for node_id, stake in stake_dict.items():
            if node_id in attacker_nodes:
                rounds_attackers.append(round_number)
                node_ids_attackers.append(node_id)
                stakes_attackers.append(stake)
            else:
                rounds_non_attackers.append(round_number)
                node_ids_non_attackers.append(node_id)
                stakes_non_attackers.append(stake)

    # Plot attackers using a different colormap
    scatter_attackers = ax.scatter(
        rounds_attackers,
        node_ids_attackers,
        stakes_attackers,
        c=stakes_attackers,
        cmap='Reds',
        marker='o',
        label='Attacker Nodes',
        alpha=0.7
    )

    # Plot non-attackers using another colormap
    scatter_non_attackers = ax.scatter(
        rounds_non_attackers,
        node_ids_non_attackers,
        stakes_non_attackers,
        c=stakes_non_attackers,
        cmap='Blues',
        marker='o',
        label='Non-Attacker Nodes',
        alpha=0.7
    )

    # Labeling the axes
    ax.set_xlabel('Round')
    ax.set_ylabel('Node ID')
    ax.set_zlabel('Stake')
    ax.set_title('Stake Evolution Over Rounds and Nodes')

def plot_aggregated_nodes_heatmap(aggregated_data: dict, winner_data: dict, validators_data: dict, num_rounds: int, num_nodes: int, fig, position: tuple, attacker_nodes: list) -> None:
    # Initialize an empty array with NaN for easier heatmap rendering
    heatmap_data = np.full((num_rounds, num_nodes), np.nan)
    
    # Populate the heatmap array with scores from the aggregated_data dictionary
    for round_number, node_data in aggregated_data.items():
        for peer_id, score in node_data:
            heatmap_data[round_number - 1, peer_id - 1] = score  # Adjust if peer IDs are 1-indexed

    # Calculate the count of NaN values per round
    missing_counts = np.sum(np.isnan(heatmap_data), axis=1)

    # Create a heatmap
    ax = fig.add_subplot(*position)
    sns.heatmap(heatmap_data, ax=ax, cmap='coolwarm', cbar_kws={'label': 'Score'},
                annot=False, fmt=".2f", annot_kws={"size": 6}, linewidths=0.5, linecolor='grey')

    # Highlight malicious columns by drawing borders
    for attacker_node in attacker_nodes:
        if attacker_node <= num_nodes:  # Ensure the node ID is within range
            ax.add_patch(plt.Rectangle((attacker_node - 1, 0), 1, num_rounds, color="red", alpha=0.3, lw=1))

    # Highlight validators and winners
    for round_number, validator_ids in validators_data.items():
        for validator_id in validator_ids:
            if round_number <= num_rounds and validator_id <= num_nodes:
                ax.add_patch(plt.Rectangle((validator_id - 1, round_number - 1), 1, 1, 
                                            color='yellow', ec='grey', lw=0.5))

    for round_number, winner_id in winner_data.items():
        if round_number <= num_rounds and winner_id <= num_nodes:
            ax.add_patch(plt.Rectangle((winner_id - 1, round_number - 1), 1, 1, 
                                        color='green', ec='grey', lw=0.5))

    # Set labels and title
    ax.set_xlabel('Node ID')
    ax.set_ylabel('Round')
    ax.set_title('Node Inclusion Scores in Aggregation Phases')

    # Update the y-axis ticks to start from 1
    ax.set_yticks(np.arange(num_rounds) + 0.5)  # Place ticks in the center of each cell
    ax.set_yticklabels(range(1, num_rounds + 1), fontsize=6)
    ax.set_xticks(np.arange(num_nodes) + 0.5)
    ax.set_xticklabels(range(1, num_nodes + 1), rotation=90, fontsize=8)
    for _, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if int(label.get_text()) in attacker_nodes:
            label.set_color('red')
        else:
            label.set_color('black')

    # Display the count of NaN values per round on the right side of the y-axis
    ax_secondary_y = ax.twinx()
    ax_secondary_y.set_yticks(np.arange(num_rounds) + 0.5)
    ax_secondary_y.set_yticklabels(missing_counts, color='black', fontsize=6)
    ax_secondary_y.set_ylim(ax.get_ylim())
    ax_secondary_y.set_ylabel('Number of ignored updates', color='black')
    ax_secondary_y.tick_params(axis='y', length=0)

if __name__ == '__main__':
    LOGGER_PATH = 'examples/committee_weights_CIFAR10_IID_global_fedavg_label_flip_33_perc/logger.log'
    GRADIENTS = False
    POS = False
    HEATMAP = True
    SAVE = True
    ATTACKER_NODES = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 55]

    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    
    # Plot each data
    acc = agg_acc_from_logger(LOGGER_PATH)
    plot_data(acc, 'Accuracy over time', 'Round', 'Accuracy', subplot=(2, 2, 1))
    
    num_of_nodes = num_of_nodes_per_round(LOGGER_PATH)
    plot_data(num_of_nodes, 'Number of nodes per round', 'Round', 'Number of nodes', subplot=(2, 2, 2))
    
    peer_trainsets = extract_peer_trainsets(LOGGER_PATH)
    peer_loss_data = extract_gradient_loss_per_round(LOGGER_PATH, gradient=GRADIENTS)
    num_epochs = 100
    plot_mean_variance_loss(peer_loss_data, num_epochs, gradient=GRADIENTS, subplot=(2, 2, 3))
    
    if POS:
        stakes_per_round = extract_stakes_per_round(LOGGER_PATH)
        plot_stakes_3d(stakes_per_round, ATTACKER_NODES, fig, position=(2, 2, 4))

    plt.tight_layout()
    plt.show()

    if SAVE:
        directory = os.path.dirname(LOGGER_PATH)
        save_path = os.path.join(directory, 'accuracy_plots.png')
        fig.savefig(save_path)
        print(f"Accuracy plots saved to {save_path}")

    if HEATMAP:
        fig = plt.figure()
        aggregated_data, winner_data, validators_data = extract_aggregated_nodes_and_scores(LOGGER_PATH)
        
        num_rounds = max(aggregated_data.keys())  # Maximum round number in data
        num_nodes = max(peer_id for round_data in aggregated_data.values() for peer_id, _ in round_data)  # Max node ID
        plot_aggregated_nodes_heatmap(aggregated_data, winner_data, validators_data, num_rounds, num_nodes, fig, position=(1, 1, 1), attacker_nodes=ATTACKER_NODES)
        plt.tight_layout()
        plt.show()
    
    if SAVE:
        directory = os.path.dirname(LOGGER_PATH)
        save_path = os.path.join(directory, 'simulation_heatmap.png')
        fig.savefig(save_path)
        print(f"Heatmap saved to {save_path}")




