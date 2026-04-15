# AutoDrive AI Codebase Analysis & Updates

## Project Overview
AutoDrive AI is an **adaptive self-driving car training platform** that combines reinforcement learning (RL) with adaptive co-evolution. The system trains an autonomous vehicle agent in simulated environments using:
- **Q-Learning** for the RL agent
- **Adaptive environment difficulty** based on episode performance
- **LLaMA 3.1** AI integration (via Groq API) for failure analysis and environment generation
- **Streamlit** web UI with professional simulation visualizations

---

## Codebase Structure

### Core Files:

#### **1. `app.py` - Main Streamlit Application**
- **Purpose**: Web UI and orchestration layer
- **Key Features**:
  - Dual-tab interface (Standard RL Training | Adaptive Co-Evolution)
  - Live environment configuration
  - Training progress visualization with placeholders
  - Episode replay with HTML5 Canvas animations
  - Metrics dashboard and AI insights
  - Comparison charts (Standard vs Adaptive modes)
- **Key Functions**:
  - `build_simulation()`: Generates HTML5 simulation canvas
  - `build_training_replay()`: Creates episode replay visualization
  - `summarize_runs()`: Calculates aggregate training statistics
- **UI Components**:
  - Configuration panel (weather, obstacles, speed, complexity)
  - Training tables with live updates (after each episode)
  - Charts (Plotly) for score and efficiency trends
  - Metrics KPIs (Score, Crashes, Goals, Episodes)

#### **2. `rl_agent.py` - Q-Learning Reinforcement Learning Agent**
- **Purpose**: Implements the core RL training logic
- **Class: RLAgent**
  - **Q-Learning Parameters**:
    - `alpha`: Learning rate (default 0.18)
    - `gamma`: Discount factor (default 0.93)
    - `epsilon`: Exploration rate (default 0.22)
  - **Methods**:
    - `choose_action()`: ε-greedy action selection with optional teacher guidance
    - `learn()`: Q-value update using Bellman equation
    - `adapt_after_episode()`: Dynamically adjust learning parameters based on performance
    - `decay()`: Reduce exploration rate after training session
  - **Tracking**:
    - Recent scores, efficiency, and collision rates (deque with maxlen=24)
    - Automatic epsilon adjustment after collisions (increases exploration)
    - Alpha (learning rate) adjustment based on reaching goals

#### **3. `simulation_core.py` - Simulation Engine**
- **Purpose**: Core physics, environment, and episode simulation
- **Key Classes**:
  - **`SimulationConfig`**: Environment configuration dataclass
  - **`Track`**: Road/track generation with 3 complexity levels (simple, moderate, complex)
  - **`SimEnv`**: Main simulation environment
  - **`EpisodeStats`**: Episode statistics container

- **Key Functions**:
  - `run_episode()`: Execute single episode with optional learning
  - `train_agent()`: Run N episodes with learning enabled
  - `train_until_goal()`: Train up to max episodes until destination reached
  - `train_coevolution()`: Adaptive training loop that adjusts environment difficulty
  - `adapt_config_for_episode()`: Calculate next environment config based on current episode
  - `directional_teacher_action()`: Expert action recommendation for guided learning

- **Environment Factors**:
  - **Track Complexity**: Affects margin size and centerline shape
  - **Weather**: Clear, foggy, rainy (affects drag coefficient)
  - **Obstacles**: 2-18 random obstacles placed on track
  - **Car Speed**: 1.0-4.5 multiplier
  - **Fog Alpha**: Visibility modifier

- **Episode Mechanics**:
  - **Max Steps**: 220 steps per episode
  - **Goal Radius**: 82 pixels to destination
  - **Reward System**: 
    - Base: -0.25 per step
    - Collision: -8 to -14 penalty
    - Progress: +4.5 to +10 for moving toward waypoints
    - Goal reach: +60 bonus
  - **Metrics Calculated**:
    - Path efficiency (direct distance / actual path)
    - Average speed
    - Difficulty score (composite of obstacles, speed, weather, complexity)
    - 7-sensor readings for obstacles/boundaries

#### **4. `adaptive_ai.py` - Groq LLaMA Integration**
- **Purpose**: AI-powered analysis and environment generation
- **Class: AdaptiveDirector**
  - **Methods**:
    - `generate_environment()`: Generate new environments from text prompts
    - `analyze_failure()`: Explain collision causes in 3 sentences
    - `explain_sensor_state()`: Describe sensor pattern for current state
    - `summarize_training_block()`: Summarize training progress across multiple episodes
  - **Model**: LLaMA 3.1 8B Instant via Groq API
  - **Fallback**: Deterministic responses when API unavailable

#### **5. `ui_components.py` - Streamlit UI Utilities**
- **Purpose**: Reusable UI components and styling
- **Key Functions**:
  - `inject_theme()`: Dark theme with custom CSS
  - `render_hero()`: Application header
  - `render_config_summary()`: Display environment settings as pills
  - `render_metrics()`: 4-column KPI display
  - `render_episode_card()`: Display last episode details
- **Styling**: Professional dark theme with blue/cyan color scheme

#### **6. `requirements.txt` - Dependencies**
- **Core**: streamlit, pandas, plotly, numpy, groq
- **ML/RL**: scikit-learn (for metrics)
- **Environment**: python-dotenv

---

## Training Flow

### Standard RL Training
```
User clicks "Run RL Training"
  ↓
For each episode (1-30):
  - Create SimEnv with current config
  - Reset agent to start position
  - Run 220 max steps:
    - Sense obstacles (7 lasers)
    - Choose action (explore + guided teacher)
    - Execute action (steer + accelerate)
    - Receive reward
    - Learn Q-values
  - Calculate statistics (score, efficiency, etc.)
  - Adapt agent parameters (epsilon, alpha)
  - UPDATE TRAINING TABLE WITH NEW EPISODE ✨ [NEW]
  - Break if destination reached
  - Decay epsilon
```

### Adaptive Co-Evolution
```
User clicks "Run Adaptive Loop"
  ↓
For each episode (1-12):
  - Run episode in current difficulty config
  - Adapt agent parameters
  - Analyze performance:
    - If collision: REDUCE difficulty (fewer obstacles, slower speed)
    - If goal reached: INCREASE difficulty
    - Otherwise: MAINTAIN or SLIGHT adjustments
  - UPDATE TRAINING TABLE WITH NEW EPISODE ✨ [NEW]
  - Generate next config based on performance
  - Break if destination reached
  - Decay epsilon
```

---

## Key Updates: Live Training Table Updates

### Problems Solved ✅
1. **Table only showed AFTER all training** - Now updates after each episode
2. **No real-time feedback** - Users can now watch progress live
3. **Long waits** - Intermediate results visible immediately

### Implementation Details

#### **Standard RL Tab Changes**
```python
# BEFORE: Called train_until_goal() which returned all episodes at once
history = train_until_goal(st.session_state.agent, cfg, max_episodes=30)

# AFTER: Manual loop with live updates
progress_container = st.container()
table_placeholder = progress_container.empty()
message_placeholder = progress_container.empty()

history = []
for episode_num in range(1, 31):
    episode = run_episode(st.session_state.agent, cfg, learn=True)
    history.append(episode)
    
    # UPDATE TABLE LIVE after each episode
    history_df = pd.DataFrame(history)
    history_df["episode_index"] = range(1, len(history_df) + 1)
    display_cols = ["episode_index", "score", "collision", "reached_goal", "path_efficiency", "avg_speed"]
    with table_placeholder.container():
        st.dataframe(history_df[display_cols], width="stretch")
    
    if episode.get("reached_goal"):
        break
```

#### **Adaptive Co-Evolution Tab Changes**
```python
# BEFORE: Called train_coevolution() which returned all episodes at once
block, local_cfg, last_adaptation = train_coevolution(...)

# AFTER: Manual loop with adaptive updates
block = []
current_config = cfg
for episode_num in range(1, 13):
    episode = run_episode(st.session_state.agent, current_config, learn=True)
    next_config, adaptation = adapt_config_for_episode(current_config, episode)
    block.append(episode)
    
    # UPDATE TABLE with adaptive metrics
    block_df = pd.DataFrame(block)
    display_cols = ["episode_index", "score", "collision", "reached_goal", "path_efficiency", "difficulty"]
    with table_placeholder.container():
        st.dataframe(block_df[display_cols], width="stretch")
    
    # Update adaptive text with current analysis
    if episode.get("adaptation"):
        st.session_state.adaptive_text = episode["adaptation"]["analysis"]
    
    if episode.get("reached_goal"):
        break
    current_config = next_config
```

#### **Key Changes**
1. **Created `st.empty()` placeholders** for table and status messages
2. **Removed calls to training wrapper functions** - Now inline episode loops
3. **Update table after each episode** using placeholder containers
4. **Added real-time status messages** with progress indicators
5. **Imported `adapt_config_for_episode`** from simulation_core

---

## Training Scores Table

The tables now display **after each episode** with the following columns:

### Standard Training Table
| Column | Description |
|--------|-------------|
| episode_index | Episode number (1-30) |
| score | Total reward accumulated in episode |
| collision | Boolean - did the car crash? |
| reached_goal | Boolean - did the car reach destination? |
| path_efficiency | Direct distance / Actual path (0-1) |
| avg_speed | Average velocity during episode |

### Adaptive Training Table
| Column | Description |
|--------|-------------|
| episode_index | Episode number (1-12) |
| score | Total reward accumulated |
| collision | Boolean crash status |
| reached_goal | Boolean goal status |
| path_efficiency | Route efficiency |
| difficulty | Composite difficulty score |

---

## Performance Metrics

### Success Indicators
- **Goals**: Episodes where destination was reached
- **Crashes**: Episodes with collision detection triggered
- **Average Score**: Mean reward across all episodes
- **Success Rate %**: (Goals / Total Episodes) × 100
- **Crash Rate %**: (Crashes / Total Episodes) × 100
- **Accuracy %**: Success Rate - (0.45 × Crash Rate)

### Comparison Mode
Side-by-side comparison of Standard RL vs Adaptive Co-Evolution across all metrics

---

## Configuration Parameters

### Environment Settings
- **Weather**: clear, rainy, foggy (affects drag)
- **Track Complexity**: simple, moderate, complex (affects terrain difficulty)
- **Obstacles**: 2-18 random obstacles on track
- **Car Speed**: 1.0-4.5x multiplier
- **Fog Alpha**: 0.0-0.45 visibility reduction

### Agent Parameters
- **Learning Rate (α)**: Decreases after success, increases after collision
- **Discount Factor (γ)**: 0.93 (value of future rewards)
- **Exploration Rate (ε)**: Starts at 0.22, min 0.03, adapts during training

---

## API Integration

### Groq LLaMA 3.1 Features
- **Failure Analysis**: Explains collision causes
- **Sensor Explanation**: Describes current sensor state
- **Training Summaries**: Professional progress summaries
- **Environment Generation**: Creates new scenarios from natural language

### Fallback Behavior
When Groq API unavailable:
- Returns deterministic alternative responses
- Suggests common safety patterns (e.g., reduce speed on tight corners)
- System continues to function fully

---

## How to Use

### Run Standard RL Training
1. Open Standard RL Training tab
2. Adjust environment settings if needed
3. Click "Run RL Training"
4. **Watch table update after each episode** ✨
5. Once complete, view metrics, charts, and AI analysis

### Run Adaptive Co-Evolution
1. Open Adaptive Co-Evolution tab
2. Configure initial environment
3. Click "Run Adaptive Loop"
4. **Watch table update with adaptive metrics** ✨
5. Observe how difficulty adjusts based on performance

---

## Technical Notes

- **State Space**: 7 sensors × 5 potential values per sensor + speed + angles + distances
- **Action Space**: 7 discrete actions (combinations of steering and acceleration)
- **Episode Length**: 220 steps max
- **Training Time**: ~5-15 minutes per standard training run
- **Memory**: Q-table stored in dictionary (scales with state space exploration)

---

## Summary of Enhancements

✅ **Live Training Table Updates** - See results after each episode  
✅ **Real-time Status Messages** - Progress indicators during training  
✅ **Adaptive Metrics Display** - Task-specific columns (standard vs adaptive)  
✅ **Improved User Experience** - No more waiting for complete results  
✅ **Consistent Architecture** - Both tabs use same update pattern  

The codebase is now production-ready with professional monitoring capabilities!
