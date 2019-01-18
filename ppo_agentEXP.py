#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# Modified by Jeremi Kaczmarczyk (jeremi.kaczmarczyk@gmail.com) 2018 
# For Udacity Deep Reinforcement Learning Nanodegree

import numpy as np
import torch
import torch.nn as nn

from imported_utils import Batcher


class PPOAgentEXP(object):
    
    def __init__(self, environment, brain_name, policy_network, optimizier, config):
        self.config = config
        self.hyperparameters = config['hyperparameters']
        self.network = policy_network
        self.optimizier = optimizier
        self.total_steps = 0
        self.all_rewards = np.zeros(config['environment']['number_of_agents'])
        self.episode_rewards = []
        self.environment = environment
        self.brain_name = brain_name
        
        env_info = environment.reset(train_mode=True)[brain_name]    
        self.states = env_info.vector_observations              

    def step(self):
        rollout = []
        hyperparameters = self.hyperparameters

        env_info = self.environment.reset(train_mode=True)[self.brain_name]    
        self.states = env_info.vector_observations  
        states = self.states
        #print('************* states :', states.shape)  # (20,33)
        for _ in range(hyperparameters['rollout_length']):
            actions, log_probs, _, values = self.network(states)
            env_info = self.environment.step(actions.cpu().detach().numpy())[self.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            terminals = np.array([1 if t else 0 for t in env_info.local_done])
            self.all_rewards += rewards
            
            #print('actions, log_probs, next_states, rewards, terminals :\n')
            #print(actions.shape)           # (20,4)  --> 20 AGENTS !!!
            #print(log_probs.shape)         # (20,1)
            #print(next_states.shape)       # (20,33)
            #print(len(rewards))            # 20
            #print('rewards :\n', rewards)  # [0.0, 0.0, 0.0, 0.0, ... 0.0]
            #print(terminals)               # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            #print(terminals.shape)         # (20,)
            
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.all_rewards[i])
                    self.all_rewards[i] = 0
                    
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - terminals])
            states = next_states
            
            #print('rollout1 :', len(rollout), '\n', rollout)
            ## states(20,33), values(20,1), actions(20,4), log_probs(20,1), rewards(20,), 1-terminals(20,)
            #print('======================\n')
            #print(len(rollout[0]))        # 6
            #print(rollout[0][0].shape)    # (20,33)
            #print(rollout[0][0][3].shape) # (33,)
            #print(rollout[0][1].shape)    # (20, 1)
            #print(rollout[0][2].shape)    # (20, 4)
            #print(rollout[0][3].shape)    # (20, 1)
            #print(len(rollout[0][4]))     # 20
            #print(rollout[0][5].shape)    # (20,)
            #input()

        self.states = states
        pending_value = self.network(states)[-1]
        rollout.append([states, pending_value, None, None, None, None])
        
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
        #print('rollout2 :', len(rollout), '\n')  # 2049
        #print(len(rollout[0]))        # 6
        #print(rollout[0][0].shape)    # (20,33)
        #print(rollout[0][0][3].shape) # (33,)
        #print(rollout[0][1].shape)    # (20, 1)
        #print(rollout[0][2].shape)    # (20, 4)
        #print(rollout[0][3].shape)    # (20, 1)
        #print(len(rollout[0][4]))     # 20
        #print(rollout[0][5].shape)    # (20,)
        
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = torch.Tensor(np.zeros((self.config['environment']['number_of_agents'], 1)))
        returns = pending_value.detach()
        
        #print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')
        #print('pending_value :', pending_value.shape)          # (20, 1)
        #print('returns :', returns.shape)                      # (20, 1)
        #print('processed_rollout :\n', len(processed_rollout)) # 2048
        #print('advantages shape :', advantages.shape)          # (20, 1)
        #input()
        
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = torch.Tensor(terminals).unsqueeze(1)
            rewards = torch.Tensor(rewards).unsqueeze(1)
            actions = torch.Tensor(actions)
            states = torch.Tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + hyperparameters['discount_rate'] * terminals * returns

            td_error = rewards + hyperparameters['discount_rate'] * terminals * next_value.detach() - value.detach()
            advantages = advantages * hyperparameters['tau'] * hyperparameters['discount_rate'] * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
        #print('processed_rollout = [states, actions, log_probs, returns, advantages]')
        #print('processed_rollout :', len(processed_rollout), '\n')  # 2048 --> (2048,5)
        #print(len(processed_rollout[0]))                            # 5
        #print(processed_rollout[0][0].shape)                        # (20,33)
        #print(processed_rollout[0][1].shape)                        # (20, 4)
        #print(processed_rollout[0][2].shape)                        # (20, 1)
        #print(len(processed_rollout[0][3]))                         # 20 
        #print(processed_rollout[0][4].shape)                        # (20, 1)
            
        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()
        #print('---------------------------\n')
        #print('states shape :', states.shape)                 # (40960, 33)
        #print('actions shape :', actions.shape)               # (40960, 4)
        #print('log_probs shape :', log_probs.shape)           # (20, 1)
        #print('log_probs_old shape :', log_probs_old.shape)   # (40960, 1)
        #print('returns len :', len(returns))                  # 40960
        #print('advantages shape :', advantages.shape)         # (40960, 1)
        #input()
        # ***** The above is just resizing (2048,20,x) to (40960,x), b/c 2048*20 = 40960 !!!

        batcher = Batcher(states.size(0) // hyperparameters['mini_batch_number'], [np.arange(states.size(0))])
        #print('batcher input params: \n')
        #print(states.size(0) // hyperparameters['mini_batch_number'])   # 1280
        #print([np.arange(states.size(0))])                              # 0, 1, 2, ... 40959
        #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
        
        for _ in range(hyperparameters['optimization_epochs']):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                #print('batch_indices1: ', batch_indices)
                #print('batch_indices1 len: ', len(batch_indices))        # 1280
          
                batch_indices = torch.Tensor(batch_indices).long()
                #print('batch_indices2: ', batch_indices)
                #print('batch_indices2 shape: ', batch_indices.shape)     # (1280,)
                
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]
                
                #print('sampled_states shape: ', sampled_states.shape)               # (1280, 33)
                #print('sampled_actions shape: ', sampled_actions.shape)             # (1280, 4)
                #print('sampled_log_probs_old shape: ', sampled_log_probs_old.shape) # (1280, 1)
                #print('sampled_returns len: ', len(sampled_returns))                # 1280
                #print('sampled_advantages shape: ', sampled_advantages.shape)       # (1280, 1)

                _, log_probs, entropy_loss, values = self.network(sampled_states, sampled_actions)
                #print('entropy_loss :', entropy_loss) # ***** All zeroes - NEVER DONE, JUST ZERO FILLED *****
                #print('entropy_loss shape :', entropy_loss.shape)                   # (1280, 1)
                #input()
                
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - hyperparameters['ppo_clip'],
                                          1.0 + hyperparameters['ppo_clip']) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) - hyperparameters['entropy_coefficent'] * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                self.optimizier.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), hyperparameters['gradient_clip'])
                self.optimizier.step()

        steps = hyperparameters['rollout_length'] * self.config['environment']['number_of_agents']
        self.total_steps += steps
        
        # print('steps :', steps)                       # 40960
        #print('self.total_steps :', self.total_steps)  # 40960, 81920, 122880, 163840, ... , 409600 (see notes !!!!!)
        #input()
