Track 1: "Can It Play DOOM?"
Domain: Deep Reinforcement Learning, Spatial Intelligence & Autonomous Control
Environment: DOOM (via ViZDoom Engine)

1. Mission Brief
For decades, "Can it run DOOM?" has been the ultimate test for hardware. In this hackathon, we flip the question to Machine Learning: "Can your neural network actually play DOOM?"

Standard RL bots fail the moment a wall shifts or an enemy spawns in a new corner because they simply memorize fixed button sequences. Your challenge is to design and train an autonomous agent capable of playing through DOOM levels zero-shot—surviving demon ambushes, navigating complex multi-room architectures, handling doors and keycards, and reaching the level exit.

No static dataset is provided. Your agent must learn by interacting directly with the game engine.

Compute & Sensory Guidance: Training solely on raw RGB screen pixels requires heavy GPU compute and spends 90% of training time just learning basic edge detection. Instead, we strongly recommend utilizing ego-centric spatial data (extracting internal sector geometry, depth, and visible entity telemetry directly from the engine), which runs at thousands of ticks per second on a standard CPU. However, if your architecture benefits from a hybrid approach combining pixel buffers with spatial telemetry, you are completely free to combine them.

2. Core Challenge Objectives
Your model must be generalizable across different maps and unseen layouts rather than overfitted to a single scenario. A successful agent must contend with:

Partial Observability: Navigating 3D spaces without a god-view radar, relying only on what is currently within the player's field of view and local sensory range.
Long-Horizon Progression: Solving multi-step room dependencies—such as exploring branches to find a Keycard, remembering where a Locked Door was sighted earlier, and backtracking to unlock the exit.
Tactical Combat & Survival: Managing health, ammunition, and movement momentum while dodging incoming projectiles and prioritizing multiple threats.

3. Deliverables & System Architecture
Your final submission must include your trained agent, training pipeline, and inference setup covering the following core areas:

Sensory & State Representation: Your observation extraction pipeline from the DOOM engine (ego-centric spatial rays, entity vectors, depth buffers, or hybrid vision-spatial inputs) designed so the agent cannot simply memorize map coordinates.
Model Architecture & Memory: Your custom neural network topology, specifically addressing how the agent maintains spatial memory across rooms and handles variable numbers of enemies.
Training & Exploration Pipeline: Your reinforcement learning implementation, custom reward shaping, and exploration mechanisms that drive the agent to navigate unfamiliar rooms rather than getting stuck in corners.
Note on Innovation: Innovative, custom-engineered architectures are heavily rewarded. Simply importing an off-the-shelf, plug-and-play Stable-Baselines3 PPO script with default hyperparameters will score poorly. We are looking for bespoke neural modules, custom spatial memory systems, intrinsic curiosity mechanisms, or hierarchical control policies.

4. Prerequisites & Recommended Libraries
Core Libraries
vizdoom: The official Python API binding for the DOOM engine (pip install vizdoom), supporting both raw screen buffers and internal object/sector state extraction.
gymnasium: Standard interface for wrapping the environment step, action, and observation loops.
torch / jax + flax: For building custom spatial encoders, attention heads, recurrent memory cells, and custom RL loss functions from scratch.
Key Concepts to Research
Ego-Centric Spatial Encoding: Raycasting, local coordinate frame transformations, and Line-of-Sight filtering.
Spatial Memory & Exploration: Recurrent policies (LSTM/GRU), differentiable occupancy grids, and intrinsic curiosity rewards.
Generalization in RL: Curriculum learning, domain randomization, and hierarchical sub-goal policies.

5. Ground Rules
Zero Hardcoded Scripts: Final agents must make decisions via trained neural parameters. Hardcoded if/else rule trees, classical pathfinders (A*, Dijkstra), or trigonometry aimbots (atan2) are strictly prohibited.
Engine Modifications & Curriculum Training: You are completely free to modify the DOOM engine, map scripts (ACS), or game variables during training for focused pretraining or curriculum learning (e.g., making doors open automatically, granting infinite ammo, or isolating combat/navigation phases). However, final evaluation will run strictly on the standard, unmodified DOOM engine and vanilla game rules—no mods allowed.
Original Work: Your repository must be created during the hackathon. Pre-existing open-source libraries are permitted, provided your core agent architecture and training pipeline are built during the event.
Zero-Shot Live Evaluation: During final judging, your trained models will be tested live on unseen evaluation maps.
