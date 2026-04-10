import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

env = gym.make("HalfCheetah-v4") # This is the environment we will be using, it has an action space of 6 depicting each torque controlled for the animal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256): # Pass in the dimension of the state and the action, and then make a hidden layer of dimension 64

        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())
        
        # This function Sequential is essentially used to do a few actions back to back, so if we were to pass a state into self.shared, it would go through the first layer of 
        # state_dim to hidden_dim then through ReLU then hidden_dim to hidden_dim then through ReLU. Because we need an action (actor network) and a value (critic network) we must
        # diverge from here and put the state through 2 different layers at the end, but until now the critic and actor network can share the same layers. 
        
        self.actor_mean = nn.Linear(hidden_dim, action_dim)   # Outputs this [0.3, -0.5, 0.1, 0.8, -0.2, 0.4]
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Looks like this [0, 0, 0, 0, 0, 0]

        # From here, we put the state through self.actor_mean in order to get a mean for the action space, because we are doing continuous actions, so we need to sample from a 
        # Gaussian distribution (Normal dist), so we calculate the mean and then we calculate log_std, the standard deviation but we do log_std to ensure that it is always positive
        # and initially we set it to torch.zeros because std is what controls the exploration and exploitation, a higher std means more exploration because the distribution is wider.

        self.critic = nn.Linear(hidden_dim, 1)

        # This is the value which is just the expected cumulative reward and it used to calculate the advantage (which is used for loss minimization) and also is the output of the critic network.

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        # Optimizer is a fancier way to do gradient ascent that is not just the generic gradient step as this specific one tracks momentum and adapts the learning rate.

        self.k_epochs = 3

        # This is the number of times we run the entire learning loop on a single batch

    def forward(self, state):

        x = self.shared(state)

        # First define x to be the input that goes into the actor and critic network after the initial shared layers.

        mean = self.actor_mean(x) 
        std = torch.exp(self.log_std) # Looks like this now [1, 1, 1, 1, 1, 1]
        dist = torch.distributions.Normal(mean, std)

        # Then we calculate the mean at a certain state just by putting it through the actor_mean layer that we defined earlier.
        # Then we calculate the std which is just the standard deviation from that state which as said earlier is a factor for exploration/exploitation, and here we are
        # exponentiating it because it is in log form, which now ensures it is always positive.

        # Then we identify the distribution by using torch.distribtutions.Normal to make a Normal distribution based on the mean and std values

        action = dist.sample()

        # We sample the action from this distribution to create the action vector that will be applied to each of the joints

        value = self.critic(x)

        # We make the value equal to the state passed through the critic network

        return dist, action, value, x

    def rollout(self, env, timesteps = 2048):

        actions = [] # Compute the new log_probs for the ratio
        values = [] # To compute advantages 
        states = [] # We reevalute these states with the new policy
        rewards = [] # Used for RTG
        log_probs = [] # Used for ratio computation
        dones = [] # See if the episode ended at each step so we do not overcalculate the returns
        activations = [] # All the activations to put through the SAE

        # The point of the rollout is to collect data from the current policy and how it interacts with the environment in order to calculate things like loss and how it will
        # adjust the next policy iteration.

        state, _ = env.reset()

        # We are just doing this to initialize the first state at the start of the environment.

        for i in range(timesteps):

            state_tensor = torch.FloatTensor(state)

            # State is initially a numpy array so we have to convert it into a tensor so we can use the torch methods

            dist, action, value, activation_tensor = self.forward(state_tensor)
            log_prob = dist.log_prob(action).sum()
            activations.append(activation_tensor.detach()) 

            # First we get the distribution, action and value from the forward function we made earlier
            # Then in order toget the log prob of the action vector we have, we have to take the log of the product of all the elements in the vector which is just equal to
            # the sum of them because log(a*b) = log(a) + log(b)

            next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())
            
            # After we got the action, we make it back into a numpy array because it is currently a tensor and then we do env.step in order to move in the environment.
            # This outputs the next_state, a reward, whether the episode was terminated(due to natural occurence) or truncated(ended due to time limit) and info

            done = terminated or truncated

            # Define done to be either terminated or truncated because either way the episode ended and done is now a boolean

            states.append(state)

            actions.append(action.detach())
            values.append(value.detach())
            log_probs.append(log_prob.detach())

            rewards.append(reward)
            dones.append(done)

            # Add all the information to their respective lists storing all the info, and use the detach function again which basically removes the tesnor from the computational
            # graph so it can't be changed while computing things.

            if done:
                state, _ = env.reset()
            else:
                state = next_state

            # If the done variable is true, then we just reset the environment again but if its false we make the state = to the next state that was produce from taking our action
            # in the environment. Then we return all the repsective lists of data we stored.

        return states, actions, log_probs, rewards, values, dones, activations

    def compute_rtgs(self, rewards, dones, gamma=0.99):

        RTG = []

        # THe goal of this function is to calculate the rewards-to-go a list of total future cumulative reward for being at a given state.

        future_return = 0
        for t in reversed(range(len(rewards))):

            if dones[t]:
                future_return = 0
            future_return = rewards[t] + gamma * future_return
            RTG.append(future_return)

        # First we define future return to be 0 because we are starting from the end of the list, if at timestep t done is equal to True that means the episode ended so we have to set 
        # future return to 0 again because nothing comes after it. If done is False that means the epsiode is not over so we calculate future return to be the reward at that timestep
        # plus gamma * the future return, gamma being some value that discounts later rewards because earlier rewards are more important. Then we just append that value to the RTG list 
        # and reverse the list because we are undoing the backwards order. 

        # Formula looks like this rtg[t] = reward[t] + gamma * rtg[t+1] and if done[t] = True then rtg[t] = reward[t]

        RTG.reverse()
        return RTG

    def learn(self, states, actions, old_log_probs, returns, values, clip_epsilon = 0.2):

        # The core of the PPO algorithm where the agent learns and we also implement the clipping technique of PPO that makes it unique. Typical value for clipping epsilon
        # is around 0.2, and we are taking in all the lists from the rollout including log_probs because that is required to the ratio thats used for our actor_loss.

        returns = torch.FloatTensor(returns)
        values = torch.stack(values).detach()
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs)

        # Returns is currently in the form of a list, so in order to convert into a tensor we use FloatTensor because we want to do calculations with the values tensor. 
        # Values is equal to torch.stackk(values).detach() we do .stack in order to convert a list of tensors into one tensor of a different dimension, and by doing .detach()
        # we remove the gradient connections it has so that the old values are not rewritten, allowing the backprop to correctly calculate error based on the value at that specific
        # time. States_tensor is again converting a list of states into a tensor to put it through the network and actions_tensor/old_log_probs are lists of tensors combined into 
        # one tensor so that computation is easier. 

        for step in range(self.k_epochs):

            advantages = returns - values

            # Advantage is calculated as the difference between the returns and the values because we want to see how much better or worse was this specific action than expected.
            # Then we normalize the advantages tensor so the gradients are not huge values and also create a balance of positive (reinforce) and negative (discourage) advantages.

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            dist, _, new_values, _ = self.forward(states_tensor)

            # Getting the distribution and new_values by running the state tensor back into the network with the current weights so we can get the action distirbution needed for 
            # getting the log_probs and the new_values (the new estimated critic values needed for the critic loss). Then we do new_values.squeeze() because we want to match
            # the dimensions of the returns tensor in order to calculate the critic loss.

            new_values = new_values.squeeze()

            new_log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # We get the new_log_probs by taking the log_prob of each action dimension and then summing it to get a total log prob for each timestep. 
            # Then we are computing the ratio which is important for the PPO step because we are seeing likely an action is to be taken. Greater than 1 meaning more likely
            # and we are calculating it using exp(new - old) because when we do it through logs this is equivvalent to new/old.

            unclipped = ratio * advantages 
            clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            actor_loss = -torch.min(clipped, unclipped).mean()

            # Unclipped is seeing the naive approach to the loss, so its how much we should reinforce or discourage this action based on how good it was (advantage) scaled by
            # how much the polcy changed (ratio). Clipped is doing the same thing but it keeps the value within some threshold so that the loss does not change the policy too 
            # much in one update. Then the actor loss takes the min of these two because we never want to overshoot the goods and bads, meaning if an action is good we do not 
            # want to reinforce too much and if an action is bad we do not want to discourage too much because ultimately PPO is a conservative algorithm making training more stable.
            # The reason we have the negative sign is because normally for loss, we are doing gradient descnet which is what torch does, but in this scenario we are not trying 
            # to decrease the loss in a loss landscape, we are trying to increase the reward in a reward landscape so we want to find peaks which is why we reverse the descent.

            critic_loss = ((returns - new_values)**2).mean()

            # Critic loss is merely equal to the mean of the squares of the difference between the actual returns versus the values that were predicted by the critic network.

            total_loss = actor_loss + critic_loss
            self.optimizer.zero_grad()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5) # Do this to prevent policy collapse and for stability
            self.optimizer.step()

            # Total loss is equal to both the losses combined because the layers they start with is initially the same so we can combine to do one backprob pass.
            # Then we do optiimizer.zero_grad to reset the gradients to zero so that they do not accumulate and get added to newer gradients. Then we do total_loss.backward()
            # so it computes gradients for all parameters and then we step the optimizer so that it can continue the gradient descent.

model = ActorCritic(state_dim=17, action_dim=6)

for iteration in range(200):

    states, actions, log_probs, rewards, values, dones, activations = model.rollout(env)
    returns = model.compute_rtgs(rewards, dones)
    model.learn(states, actions, log_probs, returns, values)
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, total reward: {sum(rewards)}")

all_activations = []
all_states = []
for _ in range(10):  # Collect from trained policy
    states, _, _, _, _, _, activations = model.rollout(env)
    all_activations.extend(activations)
    all_states.extend(states)

activations_tensor = torch.stack(all_activations)

torch.save(activations_tensor, 'activations.pt')
torch.save(torch.FloatTensor(np.array(all_states)), 'states.pt')

print(f"Saved activations: {activations_tensor.shape}")









                






