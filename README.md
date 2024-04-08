# DQN-Maze-Solver

Project 2: DQN
DQN is an extension of Q-learning. Q-learning is an approach for an agent to maximize reward in the long run by determining which actions yield the greatest results at all states. This works very well with a limited number of states and actions but becomes extremely computationally expensive very quickly as the number of states and actions increases as the table must be (s x a) in size and is constantly updated. DQN approximates these table values which means computation time is greater as a baseline but scales more slowly than the (s x a) factor which regular Q learning does. The following diagram depicts Q-learning and DQN and their similarities:

![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/dde21e74-d5ff-453c-8853-4939419702bf)


Fig 1: Q-learning vs. DQN (https://www.geeksforgeeks.org/deep-q-learning/)

The DQN training cycle works very similarly to Q-learning. Below are the steps required to run a DQN system:
	Initialize a Q-network with random weights
	Create and experience replay buffer to store transitions
	Episode begins
	Reset the environment and pick an initial state (start closer to the goal at the beginning)
	Step beings
	Select an action using epsilon-greedy based on the Q-network
	Execute the action
	Store the transition in the experience replay buffer
	If enough samples are present
	Sample a batch
	Update the Q-network by minimizing Qloss
	If s’ is terminal set target = R
	Otherwise:  Target=R+γ  max┬a'⁡〖Q_target (s^',a^')〗
	Repeat

Double Q-Learning builds upon Q-Learning by introducing an additional step and utilizing two separate Q-value estimators (or networks in the context of Deep Double Q-Learning). This structure allows for a more nuanced approach to estimating future rewards, addressing the common issue of overestimation bias found in traditional Q-Learning. This approach enables a more thorough analysis of the environment by considering additional perspectives on action outcomes. Furthermore, by periodically averaging the values between these networks, Double Q-Learning aims to achieve more consistent and reliable results across trials. The technique not only deepens the understanding of potential future states by examining them through the lens of two separate evaluators but also stabilizes learning by blending their insights, thereby enhancing the agent's ability to devise strategies that are both effective and grounded in a more balanced assessment of the environment's dynamics.

 ![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/75e0979c-61a2-4885-97c6-0bcb8cb3be5c)

Fig 2: Double Q: (https://rubikscode.net/2021/07/20/introduction-to-double-q-learning/)
Prioritized Q-learning builds upon Q-learning as well but focuses on improving the experience replay rather than the Q-learning process itself. Standard experience replay samples experiences uniformly at random, treating each experience as equally important for learning. Prioritized Q-learning introduces a method to prioritize experiences based on their importance, measured by the temporal difference (TD) error. Experiences with higher TD errors, indicating a larger discrepancy between the expected and actual Q-values, are deemed more significant for learning, and are sampled more frequently. This prioritization allows the learning process to focus on experiences from which the agent can learn the most, accelerating the learning efficiency and ideally leading to faster convergence on optimal policies. By dynamically adjusting the sampling probability of experiences based on their contribution to learning, Prioritized Q-Learning seeks to make more effective use of the experience replay buffer, optimizing the learning process by concentrating on the most informative experiences.

Before I display the results, I have a few notes on my training methodology. I used a variable epsilon that changed between epochs like so:

 ![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/a9cb86f3-13b2-45c8-83e9-9abeebb80e91)

Fig 3: Variable epsilon
This allowed for a lot of exploration early on and greater accuracy later in training. 
I tried a few different gamma values but everything below 0.9 yielded very incorrect solutions to the maze. As a result all of my testing was done with a gamma of 0.95.
I added a cutoff for the number of steps each episode would take before ending. This was added to save time as training would never finish if the agent continued to run into the wall forever. If the agent had a total reward less than –(number of states) the game ended with a loss. 
The replay buffer and batch sizes were set to 10000 and 24 respectively. 24 was picked due to memory constraints on my computer. The number of epochs was set to 20000. This number was determined experimentally. 15000 probably would have been fine as well. 
I decided to also analyze how different types of networks affected training. I compared one network that simply contained 3 fully connected layers with a ReLu activation layer with another network that had two convolutional layers before the 3 fully connected layers. The implementation of this was very simple and I thought it would be an interesting comparison to make.
The reward structures were slightly different between the dynamic programming Q-learning approach and the deep learning approaches. With all of these factors in mind, below is my comparison between Dynamic Programming Q-learning, Simple DQN, Prioritized DQN, and Double DQN. 
![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/dd5e6092-e448-49a3-9a43-457c46eee027)
![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/10cc3ce6-422f-490a-9a04-88b04ee5e3ed)

Fig 4: Dynamic Programming Q-learning avg accumulated reward and policy.
![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/1dbc1a55-7cd4-49c9-bbdf-a289d7bf3473)
![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/e3d68f17-ed34-47c8-8be7-aa816f54cafa)

Fig 5: Deep Q learning with a fully connected net and normal experience replay avg accumulated reward and policy.

![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/36a33b0d-a107-4d70-bfd6-a311c9dd4576)
![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/170f829e-b386-4885-9c76-bb13db5e5149)


Fig 6: Deep Q learning with a convolutional net and normal experience replay avg accumulated reward and policy.
  ![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/5e10c1ac-43bd-4c79-be2d-ecb21dc4233a)
![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/89fbfea5-94c2-4e64-8a05-3e4847f3f886)

Fig 7: Deep Q learning with a fully connected net and priority experience replay avg accumulated reward and policy.
![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/97042aae-03de-4df8-a38b-f6dffd073b12)
![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/cc706894-ea14-41d2-af60-d30a1695b265)

  
Fig 8: Deep Q learning with a convolutional net and priority experience replay avg accumulated reward and policy.
  ![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/b1a4cf15-11cd-49b1-bbc9-1584d04c2b48)
  ![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/3a9fc2a4-7b8e-49f0-9666-38b4867cfb04)


Fig 9: Double Q learning with a fully connected net avg accumulated reward and policy.
  ![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/91bde40e-7702-490d-bb49-70856b787fba)
![image](https://github.com/emandel2630/DQN-Maze-Solver/assets/91342800/d9039660-817d-466f-a71c-24f0a3ef6ad4)

Fig 10: Double Q learning with a convolutional net avg accumulated reward and policy.

	All methods tested were able to reach the goal from every position on the board. The routes from the start to the finish line varied by method. 
	The dynamic programming approach that had no deep learning component reached its stable state at episode 200 of 1000. This value cannot be reasonably compared in speed to the epoch measure used by the remaining trials as a one-to-one comparison is not meaningful. 
The fully connected net with normal buffer DQN reached its steady state at approximately 7000 epochs. The convolutional net with normal buffer DQN reached its steady state at approximately 7500 epochs. The fully connected net with priority buffer DQN reached its steady state at approximately 7500 epochs. The convolutional net with priority buffer DQN reached its steady state at approximately 7000 epochs. The convolutional net double DQN reached its steady state at approximately 5500 epochs. The fully connected net double DQN reached its steady state at approximately 15000 epochs. 
These results yield a few different conclusions: 
	For this particular maze problem, the type of DQN approach taken did not affect the maximum average accumulated reward the agent collected at the end of training. All steady state rewards were identical.  
	The usage of convolutional vs fully connected models did NOT affect time to steady state meaningfully for DQN and priority DQN 
	The usage of convolutional vs fully connected models DID affect time to steady state meaningfully Double Q learning (5500 vs 15000 epochs) 
	Deep learning models did not accurately determine the best move at each position as bumps and oil slicks were often chosen over more efficient routes.
	There was no meaningful difference in performance between priority Q learning and a regular deep Q net 
	For this specific problem, standard Q-learning was far more effective yielding greater accuracy with far less computation. 
The results above show that although Deep Q learning in its various may be more effective on problems with a far greater number of actions and states, it is computationally inefficient on small problems such as this one. More complex systems would benefit from usage of DQNs as they scale more favorably when compared to their non-deep counterparts. 

Acknowledgements:
Huge thank you to GitHub user giorgionicoletti for the open-source framework for maze solving DQNs. 

https://github.com/giorgionicoletti/deep_Q_learning_maze
