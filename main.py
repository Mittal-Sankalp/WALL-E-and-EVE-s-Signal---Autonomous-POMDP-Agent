import json
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class WalleEnvironment:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.grid_size = self.config['map_size']
        self.pos = list(self.config['walle_start'])
        self.eve_pos = self.config['eve_location']
        self.clues = self.config['clue_locations']
        
        budget = self.config['resource_budget']
        self.battery = budget['battery']
        self.max_battery = budget['battery']
        self.time = budget['time_limit']
        self.costs = budget['action_costs']

    def step(self, action):
        cost = 0
        observation = "none"
        found_eve = False

        if action.startswith("move"):
            cost = self.costs['move']
            if action == "move_up" and self.pos[1] < self.grid_size[1] - 1: self.pos[1] += 1
            elif action == "move_down" and self.pos[1] > 0: self.pos[1] -= 1
            elif action == "move_right" and self.pos[0] < self.grid_size[0] - 1: self.pos[0] += 1
            elif action == "move_left" and self.pos[0] > 0: self.pos[0] -= 1
            
        elif action == "scan":
            cost = self.costs['scan']
            pos_key = f"{self.pos[0]},{self.pos[1]}"
            if pos_key in self.clues:
                observation = self.clues[pos_key]
            else:
                observation = "no_signal"
                
        elif action == "recharge":
            cost = self.costs['recharge'] 
            
        self.battery -= cost
        self.battery = min(self.battery, self.max_battery)
        self.time -= 1
        
        if self.pos == self.eve_pos:
            found_eve = True
            observation = "EVE_LOCATED"

        done = self.battery <= 0 or self.time <= 0 or found_eve
        
        return observation, self.pos, done, found_eve

class WalleAgent:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.grid_size = config['map_size']
        
        self.belief = [[1.0 for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        self.evidence_gathered = []
        self.trace_data = []
        self.step_count = 0
        self.path = [list(config['walle_start'])]
        
        # Apply initial signal reading to belief prior
        initial_signal = config.get('initial_signal')
        if initial_signal:
            direction = initial_signal.get('direction', '')
            strength = initial_signal.get('strength', '')
            self.evidence_gathered.append(f"Initial Intel: {strength} signal detected from {direction}")
            
            mid_x = self.grid_size[0] / 2.0
            mid_y = self.grid_size[1] / 2.0
            
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    # Bias the probabilities based on quadrant direction
                    if direction == "NE" and i >= mid_x and j >= mid_y:
                        self.belief[i][j] *= 3.0
                    elif direction == "NW" and i < mid_x and j >= mid_y:
                        self.belief[i][j] *= 3.0
                    elif direction == "SE" and i >= mid_x and j < mid_y:
                        self.belief[i][j] *= 3.0
                    elif direction == "SW" and i < mid_x and j < mid_y:
                        self.belief[i][j] *= 3.0
                        
        self._normalize_belief()
        
    def update_belief(self, pos, observation):
        x, y = pos
        if observation == "none": return
        
        if observation != "no_signal" and observation != "EVE_LOCATED":
            self.evidence_gathered.append(f"Found {observation} at [{x}, {y}]")
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                dist = abs(i - x) + abs(j - y)
                
                if observation == "signal_very_strong":
                    prob_multiplier = 2.0 if dist <= 1 else 0.1
                elif observation == "signal_strong":
                    prob_multiplier = 1.5 if dist <= 2 else 0.5
                elif observation == "signal_weak":
                    prob_multiplier = 0.5 if dist <= 2 else 1.2
                else: 
                    prob_multiplier = 0.1 if dist == 0 else 1.0
                    
                self.belief[i][j] *= prob_multiplier
                
        self._normalize_belief()

    def _normalize_belief(self):
        total_prob = sum(sum(row) for row in self.belief)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.belief[i][j] /= total_prob

    def choose_action(self, pos, battery):
        if battery < 25:
            return "recharge", f"Battery critical ({battery}%). Safety override to recharge."
            
        if random.random() < 0.3:
            return "scan", "Random exploration scan to gather environmental data."
            
        max_prob = -1
        target = pos
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.belief[i][j] > max_prob:
                    max_prob = self.belief[i][j]
                    target = [i, j]
                    
        reasoning = f"Navigating to highest belief coord [{target[0]}, {target[1]}] (Prob: {max_prob*100:.1f}%)."
        
        if pos[0] < target[0]: return "move_right", reasoning
        if pos[0] > target[0]: return "move_left", reasoning
        if pos[1] < target[1]: return "move_up", reasoning
        if pos[1] > target[1]: return "move_down", reasoning
        
        return "scan", "Reached target coordinate. Scanning area for EVE."

    def log_step(self, action, pos, obs, battery, reasoning):
        self.step_count += 1
        self.path.append(list(pos))
        
        pos_str = f"[{pos[0]},{pos[1]}]"
        bat_str = f"{battery} %"
        
        # Exact terminal headers mapping directly to the spreadsheet columns
        self.trace_data.append({
            "STEP": self.step_count,
            "ACTION": action.upper(),
            "POSITION": pos_str,
            "OBSERVATION": obs,
            "BATTERY": bat_str,
            "REASONING": reasoning
        })
        
        # Terminal print
        print(f"| {self.step_count:<4} | {action.upper():<11} | {pos_str:<5} | {obs:<18} | {bat_str:<4} | {reasoning}")

    def export_excel(self, env, found_eve, filename="Mission_Report.xlsx"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, filename)

        df_trace = pd.DataFrame(self.trace_data)

        # Simply dump the dataframe directly into the first sheet with no styling
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df_trace.to_excel(writer, index=False)
            
            # Auto-size columns slightly so it is readable
            ws = writer.sheets['Sheet1']
            widths = {"A": 6, "B": 15, "C": 15, "D": 20, "E": 15, "F": 80}
            for col, width in widths.items():
                ws.column_dimensions[col].width = width

        print("-" * 120)
        print(f"\nMission complete. Excel report generated at: {save_path}")

    def plot_trajectory(self, env, found_eve, filename="mission_trajectory.png"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, filename)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot final belief matrix as a heatmap
        belief_matrix = np.array(self.belief).T
        cax = ax.imshow(belief_matrix, cmap='Blues', origin='lower', extent=[-0.5, self.grid_size[0]-0.5, -0.5, self.grid_size[1]-0.5], alpha=0.8)
        cbar = plt.colorbar(cax, shrink=0.8)
        cbar.set_label("Agent's Final Belief Probability", rotation=270, labelpad=15)
        
        # Draw Grid lines
        ax.set_xticks(np.arange(-0.5, self.grid_size[0], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size[1], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xticks(range(self.grid_size[0]))
        ax.set_yticks(range(self.grid_size[1]))

        # Plot path
        path_x = [p[0] for p in self.path]
        path_y = [p[1] for p in self.path]
        ax.plot(path_x, path_y, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label="WALL-E Path")
        ax.scatter(path_x, path_y, c='grey', s=20, zorder=3)

        # Plot Start, EVE, and Clues
        ax.scatter(path_x[0], path_y[0], c='orange', marker='s', s=150, edgecolor='black', zorder=4, label='Start')
        ax.scatter(env.eve_pos[0], env.eve_pos[1], c='gold', marker='*', s=350, edgecolor='black', zorder=5, label='EVE (Actual)')

        clue_plotted = False
        for clue_str, signal in env.clues.items():
            cx, cy = map(int, clue_str.split(','))
            label = 'Clues' if not clue_plotted else ""
            ax.scatter(cx, cy, c='purple', marker='X', s=100, zorder=4, label=label)
            clue_plotted = True

        ax.set_title("WALL-E Mission Trajectory & Belief Map", fontsize=14, pad=15)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"Visualization saved at: {save_path}")

    def export_text_report(self, env, found_eve, filename="mission_report.txt"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, filename)
        
        with open(save_path, 'w') as f:
            f.write("="*50 + "\n")
            f.write("FINAL MISSION REPORT\n")
            f.write("="*50 + "\n")
            f.write("Operator: Sankalp Mittal (26BIT0031)\n")
            f.write(f"Mission Status: {'EVE LOCATED SUCCESSFULLY' if found_eve else 'MISSION ABORTED (RESOURCES DEPLETED)'}\n")
            f.write(f"Final Battery: {env.battery}%\n")
            f.write(f"Total Steps Taken: {self.step_count}\n\n")
            
            f.write("Evidence Found:\n")
            if self.evidence_gathered:
                for ev in self.evidence_gathered:
                    f.write(f" - {ev}\n")
            else:
                f.write(" - None\n")
            
            f.write("\n")
            if not found_eve:
                max_p = 0
                best_loc = [0,0]
                for i in range(self.grid_size[0]):
                    for j in range(self.grid_size[1]):
                        if self.belief[i][j] > max_p:
                            max_p = self.belief[i][j]
                            best_loc = [i,j]
                f.write(f"Inferred EVE Location: [{best_loc[0]}, {best_loc[1]}] (Confidence: {max_p*100:.1f}%)\n")
                f.write("Recommended Next Step: Deploy rescue team to inferred coordinates.\n")
            else:
                f.write("Recommended Next Step: Await further instructions.\n")
                
            f.write("="*50 + "\n")
            
        print(f"Text report saved at: {save_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, 'config.json')
    
    env = WalleEnvironment(config_file)
    agent = WalleAgent(config_file)
    
    print("STARTING MISSION...")
    print("-" * 120)
    print(f"| {'STEP':<4} | {'ACTION':<11} | {'POS':<5} | {'OBSERVATION':<18} | {'BAT':<4} | {'REASONING'}")
    print("-" * 120)
    
    done = False
    found_eve = False
    
    while not done:
        action, reasoning = agent.choose_action(env.pos, env.battery)
        obs, pos, done, found_eve = env.step(action)
        agent.update_belief(pos, obs)
        agent.log_step(action, pos, obs, env.battery, reasoning)
        
    agent.export_excel(env, found_eve)
    agent.plot_trajectory(env, found_eve)
    agent.export_text_report(env, found_eve)