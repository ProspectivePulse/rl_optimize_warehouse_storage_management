# DQN Agent for Warehouse Storage Optimization (OpenAI Gym)

1. **Project Overview:** 
	- This project implements a Deep Q-Network (DQN) to solve a simulated warehouse environment using PyTorch and TensorBoard for training visualization.
	- The objective of the algorithm is to optimize the usage of storage space in the simulated warehouse environment.
	  

2. **Features:**
	
	- TensorBoard logging (episode length mean, reward mean, exploration rate, frames per second)
	- Model saving/loading for inference 

3. **Model Performance Graphs (TensorBoard):**

   	![image](https://github.com/user-attachments/assets/1c5a7408-27ce-4ad8-a873-0b6695bee22e)


   	![image](https://github.com/user-attachments/assets/8f0e34e2-4471-4249-89ef-4b77f1046f28)

	Starting from the top left image, here is an interpretation of the graphs displayed:

  	- **rollout/ep_len_mean (Episode Length Mean):** Shows the average number of steps per episode. Since, the plotted curve shows a steep initial increase followed by a plateau and then a slight dip - this suggests that the agent is learning to complete tasks efficiently (or hitting the terminal conditions quicker). On the whole, this indicates policy convergence.
     
     - **rollout/ep_reward_mean (Episode Reward Mean):** Shows the average reward per episode. The sharp increase and then a plateau - shows learning progress and then eventual performance stabilization. This indicates successful training (as long as the plateau aligns with the desired behaviour).
        
     - **rollout/exploration_rate:** Shows the epsilon decay in an epsilon-greedy policy. Since it drops from ~0.5 to 0.01 early, this confirms the exploration -> explotation shift. This suggests epsilon decay schedule was well-configured.
     
     - **time/fps (Frames Per Second):** Shows the training speed (frames per second). Since it increases and stabilizes, this indicates good training pipeline performance. Although, this is not critical for policy performance, it is helpful in profiling runs.
          
   
5. **Next Steps:**

   	- Setup Experience replay
   	- Exploration Strategy to be confirmed
   	- Create/Upload the following files:
   	   - requirements.txt
   	   - config.yaml
   	   - agent.py
   	   - train.py
   	   - evaluate.py
   	   - utils.py
   	   - dqn_model.pt
   	   - any other relevant .ipynb files
	- Upload TensorBoard logs

