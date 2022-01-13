
import time
import pickle
import gym 
import os 
import sys 
import random 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from gym import wrappers
from datetime import datetime

if not os.path.isdir('./videos'):  
    os.makedirs('videos')
    
if not os.path.isdir('./weights'):
    os.makedirs('./weights')
    
MAX_FRAMES = 50000000   # Total number of frames the agent sees 
MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
MAX_EXPERIENCES = 1000000 #500000#  in the paper it is 1 million
MIN_EXPERIENCES = 50000 #5000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 84
K = 4 #env.action_space.n = 6, we set to 4 since there are only 4 meaningfull actions 
UPDATE_FREQ = 4


def one_hot_incode(X,dimension):
    
    N = len(X) # number of samples, x in R for all x in X
    
    Y = np.zeros(shape = (N, dimension)).astype(np.float32)
    
    for i in range(N):
        Y[i, X[i]] = 1
        
    return Y
    
### define layers 

class DenseLayer(object):
    def __init__(self, M1, M2, f = tf.nn.relu, use_bias = True, layerNum = 0):
        
        self.use_bias = use_bias
        self.f = f
        # we need to use scaling using 0 and var = 1 ~N(0,1)
        # He normalization 
        W0= np.random.randn(M1, M2).astype(np.float32) * np.sqrt(2./float(M1))
        self.W = tf.Variable(initial_value = W0, name = 'W_dense_%i' % layerNum)
        
        self.params = [self.W]
        
        if self.use_bias:
            self.b = tf.Variable(initial_value = tf.zeros([M2,]), 
                                 name = 'b_dense_%i' % layerNum)
            
            self.params.append(self.b)
        
    def forward(self, X):
        Z = tf.matmul(X, self.W)
        
        if self.use_bias:
            Z += self.b
            
        return self.f(Z)
        
        
class ConvLayer(object):
    def __init__(self, mi, mo, filtersz = 4, stride = 2, f = tf.nn.relu, pad = 'VALID', add_bias = True, layerNum = 0):#SAME
        
        # mi = input feature map size
        # mo = ouput feature map size
        self.f = f
        self.add_bias = add_bias
        self.stride = stride
        self.pad = pad
        
        # also here I need to normelize properlly 
        # He normalization
        self.W = tf.Variable(initial_value = tf.random.normal(shape = [filtersz, filtersz, mi, mo],mean= 0.,
                                                              stddev = np.sqrt(2.0/(filtersz*filtersz*mi))),
                                                                 name = 'W_conv_%i' % layerNum)
        
        self.params = [self.W]
        
        if self.add_bias:
            self.b = tf.Variable(initial_value = tf.zeros(mo,), name = 'b_conv_%i' % layerNum)
            
            self.params.append(self.b)
            
    
    def forward(self, X):
        conv_out = tf.nn.conv2d(X, filters = self.W, strides = [1,self.stride, self.stride,1], padding=self.pad)
        
        if self.add_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)
            
        return self.f(conv_out)
    
    
# Transform raw images for input into neural network 
# 1) Conver to grayscale
# 2) Resize
# 3) Crop

class ImgaeTransformer:
    def __init__(self):
        pass
    
    def transform(self, X):
        # convert to grey scale
        X = tf.image.rgb_to_grayscale(X)
        
        # Crop the image 
        X = tf.image.crop_to_bounding_box(X, 34 , 0 , 160, 160) # (initial high, initial width, end_high, end_width )
        
        # image resize
        X = tf.image.resize(X, [IM_SIZE, IM_SIZE],method= 'nearest' )
        X = tf.cast(X, tf.float32) /255.0
        X = tf.squeeze(X) 
        # print(X)
        return X
    
    
# creat a new state with 4 images
def update_state(state, obs_small): 
    return np.append(state[:,:,1:], np.expand_dims(obs_small, 2), axis = 2)

class ReplayMemory:
    def __init__(self, size = MAX_EXPERIENCES, fram_height = IM_SIZE, fram_width = IM_SIZE, 
                 agent_history_length = 4, batch_size = 32):
        
        """
        Args:
            size: Integer, Number of stored transitions
            fram_height: Integer, Height of a fram of an Atari game 
            fram_width: Integer, Width of a fram of an Atari game 
            agent_history_length: Integer, Number of frames stacked together to cear a state 
            batch_szie: Integer, Number of transitions returnd in a minibatch
        """
        
        self.size = size
        self.frame_height = fram_height
        self.frame_width = fram_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        
        # Pre-allocate memory 
        self.actions = np.empty(self.size, dtype = np.int32)
        self.rewards = np.empty(self.size, dtype = np.float32)
        self.frames = np.empty(shape = (self.size, self.frame_height, self.frame_width), dtype = np.float32)
        self.terminal_flags = np.empty(self.size, dtype = np.bool)
        
        # Pre-allocate memory for the states and new_states in a minibatch 
        self.states = np.empty(
                                shape = (self.batch_size, self.agent_history_length, self.frame_height, self.frame_width),
                                dtype = np.float32)#np.uint8 <- better for save memory
        
        self.new_states = np.empty( shape = (self.batch_size, self.agent_history_length, self.frame_height, self.frame_width),
                                   dtype= np.float32)#np.uint8)
        
        self.indices = np.empty(self.batch_size, dtype=np.int32)
        
    def add_experience(self, action, frame, reward, terminal):
        
        """
        Parameters
        ----------
        action : An interger-encoded action
        frame : One grayscale fram of the game 
        reward : reward the agent received for performing an action 
        terminal : A bool stating whether the episode terminate
        -------
        """
        
        if frame.shape != (self.frame_height, self.frame_width):
            # print(self.frame_height, self.frame_width, frame.shape )
            raise ValueError('Dimension of fram is wrong!')
            
        self.actions[self.current] = action
        self.frames[self.current,...] = frame # this makes it easy to manipulate only one dimension as a time (Very nice)
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size
        
    def _get_state(self, index):
        if self.count == 0 :
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
            
        return self.frames[index - self.agent_history_length + 1 : index + 1, ...]
        
        
    # remember we define our states as circular (we don't want to cross the boundary of self.current <-- apointer 
    # where we insert a new fram, and also remember the replat buffer is circular)
    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1) #self.agent_history_length = 3
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue # it is a circular so this tuple is not a valid state
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue # we check here that we are not at the boundary of an episode
                
                break
            self.indices[i] = index # only valide states
            
            
    def get_minibatch(self): # return (s, a, r, s', done)
        """
        Returns a minibatch of self.batch.size transitions
        """
        
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')
            
        self._get_valid_indices()
        
        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1) # s
            self.new_states[i] = self._get_state(idx) # s'
            
        return np.transpose(self.states, axes = ( 0 ,2,3,1)), self.actions[self.indices], \
                self.rewards[self.indices], np.transpose(self.new_states, axes = (0,2,3,1)), \
                self.terminal_flags[self.indices] 
                ## we transpose from (N, T, H, W) ---> (N, H, W, T) T == time dimension in our case == 4!
                # remember in tensorflow the it should be --> # sampels, hight, width, channels
    
    
class DQN(object):
    def __init__(self,num_channels, image_size , K, conv_layer_sizes, hidden_layer_sizes,  scope = 'modelFunction'):
        """
        K = number of output nodes
        """

        self.K = K # number of actions output 
        self.scope = scope
        # the class of the loss 
        self.huber_func = tf.keras.losses.Huber()
        
        ## OPtimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0000625)#0.00025)#1e-5) # paper from 2017
        # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-1)#
        ### Let's build the CNN 
        self.conv_layers = []
        mi = num_channels
        count = 0
        for num_output_filter, filtersz, stride in conv_layer_sizes:
            pad = 'VALID'
            layer = ConvLayer(mi, num_output_filter, filtersz, stride, layerNum= count, pad = pad)
            mi = num_output_filter
            self.conv_layers.append(layer)
            count += 1
            if pad == 'SAME':
                p1 = (stride *(image_size/stride) - image_size + filtersz - stride) / 2
            elif pad == 'VALID':
                p1 = 0#
            image_size = np.ceil(1 + (image_size - filtersz + p1 + p1)/stride) ## just to know what is the size of the output

        ###  lets calculate the size of the input after flatten 
        M1 = int(image_size**2 * mi) ## it is a squred image 
        
        # collect all the fully connected layers 
        self.dense_layers = []
        
        # architecture hiiden_layer_sizes = [[M2, f]]
        counter = 0
        for M2, f in hidden_layer_sizes:
            
            layer = DenseLayer(M1, M2, f= f, layerNum = counter)
            self.dense_layers.append(layer)
            M1 = M2
            counter += 1
        
        # Now let's creat the last layer 
        layer = DenseLayer(M1, self.K, f = lambda x:x) # linear
        
        self.dense_layers.append(layer)
        
        
        ### collect the params 
        
        # 1) collect from conv layers
        self.trainable_params = []
        for layer in self.conv_layers:
            self.trainable_params += layer.params
            
        ## 2) collect from dense layer 
        for layer in self.dense_layers:
            self.trainable_params += layer.params
            
            
    def forward(self, Z):
        
        # convolution step 
        for layer in self.conv_layers:
            # print(Z.shape)
            Z = layer.forward(Z)
        # print(Z.shape)  
        # flatten step 
        n, w, h, c = Z.shape # number of samples in X
        Z = tf.reshape(Z, shape = (int(n), int(h*w*c) ))
        
        # fully connected
        
        for layer in self.dense_layers:
            # print(Z.shape)
            Z = layer.forward(Z)
        
        # print(Z.shape)
        # 
        return Z
         
    def predict(self, x):
        return self.forward(x)
    
    def cost(self, S, actions, G):
        prediction = self.forward(S) # S is the states, predictions is actions 
        
        # now take into account only the actions that you choose in the past 
        actual_chosen_actions = prediction * tf.one_hot(actions, self.K)
      
        selected_action_values = tf.reduce_sum(actual_chosen_actions, axis = [1])
        # print(selected_action_values, "g", G)
        costi = tf.reduce_mean( self.huber_func(y_true = G, y_pred = selected_action_values)) # in the paper it is L2 not huber
        
        return costi
    
    
    def sample_action(self, x, eps):
        if np.random.random() < eps:
            # print("hhh i am in")
            return np.random.choice(self.K)
        else:
            x = tf.expand_dims(x, axis = 0)
            # print("hallo:", x.shape)
            x = x.numpy()
            x = np.float32(x)
            # print(x)
            a_pos = self.predict(x)
            # print(np.argmax(a_pos[0]))
            return np.argmax(a_pos[0])
        
    
    def update_weights(self, states, actions, targets):
        
        
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            
            cost_i = self.cost(states, actions, targets)
        
        gradients = tape.gradient(cost_i, self.trainable_params) 
        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_params))
        
        return cost_i
    
    
    def copy_params_from(self, model):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(model.trainable_params[i].numpy())
        # self.trainable_params = [param for param in model.trainable_params] # here is the problem 
        
        
    def save(self, name ):
        params = [param.numpy() for param in self.trainable_params]
        file_name = 'weights/params' + name +'.pickle' 
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
            
        print('Saving wweights!')
            
        
    def load(self,  name ):
        file_name = 'weights/' + name +'.pickle'
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
            
        for tp,p in zip(self.trainable_params, params):
            tp.assign(p)
            
        print("weights load successfully!")
        

        
def learn(model, target_model, experience_replay_buffer, gamma, batch_size):# something with the batchsize id weird
    # SAmple experiences
    states, actions, rewards, next_states, dones = experience_replay_buffer.get_minibatch()
    
    # Calculate targets
    next_Qs = target_model.predict(next_states) # in R(n*Num_actions)
    next_Q = np.amax(next_Qs, axis = 1)
    targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q
    
    # Update model
    cost_i = model.update_weights(states, actions, targets)
    return cost_i

 

def watch_agent(env, model, image_transformer = None):
    s = env.reset()
    obs_small = image_transformer.transform(s)
    state = np.stack([obs_small] * 4 , axis = 2) #we creat the first state s0
    done = False
    episode_reward = 0
    
    while not done:
        env.render()
        action = model.sample_action(state, eps = 0.01)
        obs, reward, done, info = env.step(action)
        obs_small = image_transformer.transform(obs)
        next_state = update_state(state, obs_small) 
        
        # Compute total reward
        episode_reward += reward
        state = next_state
        time.sleep(0.02)
        
    print("Episode rewards:", episode_reward)
    env.close()
    
def record_agent(env, model,image_transformer = None, videoNum= 0):
    dire = './videos/' + 'vid_' + str(videoNum)
    env = wrappers.Monitor(env, dire, force = True)
    s = env.reset()
    obs_small = image_transformer.transform(s)
    state = np.stack([obs_small] * 4, axis = 2)
    done = False
    episode_reward = 0
    print("The agent is playing, please be patient...")
    while not done:
        action = model.sample_action(state, eps = 0.01)
        obs, reward, done, info = env.step(action)
        obs_small = image_transformer.transform(obs)
        next_state = update_state(state, obs_small)
        
        episode_reward += reward
        state = next_state
        # time.sleep(0.02)
        
    print("record video game in folder video %s / " % 'vid_' + str(videoNum), "episode reward: ", episode_reward)
    return episode_reward
                

def play_one(env, total_t, experience_replay_buffer, model, target_model, image_transformer, gamma, batch_size, epsilon):#, 
 #            epsilon_change, epsilon_min):
    
    t0 = datetime.now()
    
    # Reset the environment 
    
    obs = env.reset()
    
    # No. op_max to ensure different starting
    for _ in range(random.randint(1, 10)): #### should be 0 to 10 
        obs, _, _, info = env.step(np.random.choice(K)) 
        
        
    obs_small = image_transformer.transform(obs)
    state = np.stack([obs_small] * 4 , axis = 2) #we creat the first state s0
    # print(state.shape)
    cost = 0 
    
    total_time_training = 0 
    num_steps_in_episode = 0 
    episode_reward = 0
    
    done = False
    live_old = info['ale.lives']
    while not done:
        
        # update target network
        if total_t % TARGET_UPDATE_PERIOD == 0:
            target_model.copy_params_from(model)
            print("Copied model parameters to target network, total_t = %s, period = %s" % (total_t, TARGET_UPDATE_PERIOD))
            
        # Take action 
        action = model.sample_action(state, epsilon)
        # print("action:", action)
        
        obs, reward, done, info = env.step(action)
        live_new = info['ale.lives']
        
        if live_new  < live_old:
            done_clip = True
            live_old = live_new
        else:
            done_clip = done 
        
        # print("done_clip:", done_clip)
        obs_small = image_transformer.transform(obs)
        next_state = update_state(state, obs_small) # creat the 4-images for the next iteration
        
        # Compute total reward
        episode_reward += reward
        
        # clipping the reward 
        cliped_reward = clip_reward(reward)
        
        # Save the latest experience in the Buffer
        experience_replay_buffer.add_experience(action, obs_small, cliped_reward, done_clip)
        
        # Train the model, keep track of time 
        t0_2 = datetime.now()
        if total_t % UPDATE_FREQ == 0 : # learn 
            cost = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
        dt = datetime.now() - t0_2
        
        # More debugging info 
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1
        
        state = next_state
        
        total_t += 1
        
        #epsilon = max(epsilon - epsilon_change, epsilon_min)
        epsilon = define_epsilon(total_t, epsilon)
        if num_steps_in_episode == MAX_EPISODE_LENGTH:
            break 
        
    return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon,\
        cost
        
        
def smooth(x):
    n = len(x)
    y = np.zeros(n)
    
    for i in range(n):
        start = max(0,i-99)
        y[i] = float(x[start:(i+1)].sum()) / (i -start + 1)
# Just smooth of 100 points     
    return y 

def clip_reward(reward):
    if reward > 0:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1
    
    
    
def define_epsilon(total_t, epsilon):
    
    if total_t <= 10**6:
        epsilon_change = (1-0.1) / 10**6
        epsilon = epsilon - epsilon_change

    # elif total_t > 10**6 and total_t < 2*(10**6):
    #     epsilon_change = (0.1-0.01)/10**6
    #     epsilon = epsilon - epsilon_change
        
    else:
        epsilon = 0.1
        
    return epsilon
 

def play_regular(env, model, image_transformer):
    
    obs = env.reset()
    obs_small = image_transformer.transform(obs)
    state = np.stack([obs_small] * 4 , axis = 2) 
    

    total_reward = 0
    done = False
    
    while not done:
        action = model.sample_action(state, eps = 0.01)
        obs, r, done, info = env.step(action)
        
        small_state = image_transformer.transform(obs)
        next_state = update_state(state, small_state)
        
        total_reward += r
        state = next_state
        
        
    return total_reward
        
        

def main_train():
    choice = 'd'
    while choice != 'Y' and choice != 'N':
        print("DO you want to start training from scratch (Y/N)?:", end = " ")
        choice = input()  
        if choice not in ['N', 'Y']:
            print('please type Y for yes and N for No!')
    
    
    conv_layer_sizes =  [(32,8,4), (64, 4,2), (64,3,1)] 
    hidden_layer_sizes =[[512, tf.nn.relu]] 
    # in the paper the optimizer is RMSProp + minibatch of 32
    gamma = 0.99
    batch_sz = 32
    #num_episodes = 6500
    total_t = 0
    experience_replay_buffer = ReplayMemory()

    
    # epsilon 
    max_frames = MAX_FRAMES 
    epsilon = 1.0

    
    
    # Creat environment 
    env = gym.make("BreakoutDeterministic-v4")
    
    # creat models (model and target_model )
    # K <- possible actions 
    model = DQN(num_channels = 4, image_size = 84, K = K, conv_layer_sizes= conv_layer_sizes,
                hidden_layer_sizes = hidden_layer_sizes,scope = 'model')
    
    target_model = DQN(num_channels = 4, image_size = 84, K = K, 
                       conv_layer_sizes= conv_layer_sizes, hidden_layer_sizes = hidden_layer_sizes,
                       scope = 'Target_model')
    
    image_transformer = ImgaeTransformer()
    
    
    ### Loads weights 
    if choice == 'N':   
        try:
            model.load(name = 'paramsModel_modified_DDQN_Deterministic_v4')
            target_model.load(name = 'paramsTargetModel_modified_DDQN_Deterministic_v4')
        except:
            
            print("check if the weights are exist in the weights folder.")
            sys.exit()
            
        
        print("How many frams do you want to train?")
        total_t = 50 * (10**6)
        max_frames = int(input())
        max_frames = max_frames + total_t
        
        print("Populating experience replay buffer...")
    
        obs = env.reset()
        
        # no-op max
        for _ in range(random.randint(1, 10)): 
            obs, _, _, info = env.step(np.random.choice(K)) 
        
        obs_small = image_transformer.transform(obs)
        state = np.stack([obs_small] * 4 , axis = 2)
        live_old = info['ale.lives']
        for i in range(MIN_EXPERIENCES):
            action = model.sample_action(state, 0.05)
            
            obs, reward, done, info = env.step(action)
            cliped_reward = clip_reward(reward) # clliping 
            live_new = info['ale.lives']
            
            if live_new  < live_old:
                done_clip = True
                live_old = live_new
            else:
                done_clip = done 
                    
            obs_small = image_transformer.transform(obs)
            next_state = update_state(state, obs_small) # creat the 4-images for the next iteration
            
            
            # Save the latest experience in the Buffer
            experience_replay_buffer.add_experience(action, obs_small,  cliped_reward, done_clip)
            state = next_state
            
            if done:
                obs = env.reset()
                # no-op max
                for _ in range(random.randint(1, 10)): 
                    obs, _, _, info = env.step(np.random.choice(K))
                    
                obs_small = image_transformer.transform(obs)
                state = np.stack([obs_small] * 4 , axis = 2)
                live_old = info['ale.lives']
        
     
    else:
        epsilon = 1
       # epsilon_min = 0.1
       # epsilon_change = (epsilon - epsilon_min) / 1000000# 500000 # in the paper it is a million 
        total_t =  0 #14635372#10001114#5651834 #1912651 # we stopped here 
        # watch_agent(env, model)
        print("Populating experience replay buffer...")
        
        obs = env.reset()
        
        # no-op max
        for _ in range(random.randint(1, 10)): 
            obs, _, _, info = env.step(np.random.choice(K)) 
            
        live_old = info['ale.lives']
        for i in range(MIN_EXPERIENCES):
            action = np.random.choice(K)
            obs, reward, done, info = env.step(action)
            cliped_reward= clip_reward(reward) # clliping 
            
            live_new = info['ale.lives']
            
            if live_new  < live_old:
                done_clip = True
                live_old = live_new
            else:
                done_clip = done 

            obs_small = image_transformer.transform(obs)
            # print(obs_small.shape)
            
            experience_replay_buffer.add_experience(action, obs_small,  cliped_reward, done_clip)
            
            if done:
                obs = env.reset()
                
                # no-op max
                for _ in range(random.randint(1, 10)): 
                    obs, _, _, info = env.step(np.random.choice(K)) 
                
                live_old = info['ale.lives']

    
    # Play a number of episodes and Learn!
    t0 = datetime.now()
    print("Learning!!")
    # for i in range(num_episodes):
    i = 0
    episode_rewards= [] # np.zeros(100)
    # n2 = len(episode_rewards)
    while total_t <= max_frames:
                                                                 
        
        total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon, cost = play_one(env, total_t, experience_replay_buffer, model, target_model,
                                                                                                    image_transformer,gamma, batch_sz, epsilon)#,epsilon_change,
                                                                                                    #epsilon_min)
        # if i % 100 == 0:
        episode_rewards.append(episode_reward)
        
        last_100_avg = np.mean(episode_rewards[-100:])#episode_rewards[max(0,i-100): i+1].mean()
        
        if i % 50 == 0:
            print("Episode:", i,
                "Duration:", duration,
                "Num steps:", num_steps_in_episode,
                "Reward:", episode_reward,
                "Training time per step:", "%.3f" % time_per_step,
                "Avg Reward (Last 100):", "%.3f" % last_100_avg,
                "Epsilon:", "%.3f" % epsilon,
                "Cost: %.8f" % cost,
                "total_frames: %i" % total_t
              )
        
        if i % 1000 == 0 :
            record_agent(env, model, image_transformer=image_transformer, videoNum= str(i))
            

        sys.stdout.flush()
        
        if total_t  >= MAX_FRAMES:
            break
    
        i += 1
        
        # i = i % n2
        
    print("Total duration:", datetime.now() - t0)
        
    model.save(name = 'Model_modified_DDQN_Deterministic_v4')
    target_model.save(name = 'TargetModel_modified_DDQN_Deterministic_v4')
    # Plot the smoothed returns
    y = smooth(np.array(episode_rewards))
    plt.plot(np.array(episode_rewards), label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()


    record_agent(env, model, image_transformer=image_transformer, videoNum = i)


def main_statistic(num_of_games = 300, statistics = True, video= False):
    
    conv_layer_sizes =  [(32,8,4), (64, 4,2), (64,3,1)] 
    hidden_layer_sizes =[[512, tf.nn.relu]]
    # in the paper the optimizer is RMSProp + minibatch of 32

    # Creat environment 
    env = gym.make("BreakoutDeterministic-v4")
    
    # creat models (model and target_model )
    # K <- possible actions 
    model = DQN(num_channels = 4, image_size = 84, K = K, conv_layer_sizes= conv_layer_sizes,
                hidden_layer_sizes = hidden_layer_sizes,scope = 'model')
    

    model.load(name = 'paramsModel_modified_DDQN_Deterministic_v4')
    
    image_transformer = ImgaeTransformer()
    
    if statistics :
        rewards = []
        for i in range(num_of_games):
            
            r = play_regular(env, model, image_transformer)
            
            rewards.append(r)
            
            print("Episode %i: %.f" % (i, r))
            
        with open('rewards_stat.pickle', 'wb') as file:
            pickle.dump(rewards, file)
        
        
        plt.figure(1)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['xtick.labelsize']= 12
        plt.rcParams['ytick.labelsize']= 12
        plt.hist(rewards, bins = 10)

        plt.xlabel('Points', fontsize= 20)
        plt.ylabel('# Games', fontsize=20)
    if video:
        r = 0
        count = 0
        while r <= 428:
            r = record_agent(env, model,image_transformer = image_transformer, videoNum= 'epsilon_0.01_'+"best_game")
            
            count += 1
            
            print("reward:", r, "Episode:", count)
        
    
       
if __name__ == '__main__':
    print("Press:")
    print("1 - for training")
    print("2 - for agent statistics over several games (you'll enter the number of games later)")
    print("3 - for watching the agent play")
    print("4 - for recording the agent play")
    y = input('Enter a valid number:')
    y = int(y)
    # main for train 
    if y == 1:
        main_train()
        
    elif y == 2:
        # check statistics
        while True:
            bo = input("Do you want to save a game where the agent finish the game (Y/N)? ")
            
            if bo == 'Y' or bo == 'y' or bo == 'yes' or bo == 'Yes':
                vi = True
                break
            elif bo == 'N' or bo == 'n' or bo == 'no' or bo == 'No':
                vi = False
                break
            
            print("Please enter a valid answer Y/N!")
        num_of_games = int(input("Enter the number of game that you want to test on:"))
        main_statistic(num_of_games = num_of_games, statistics = True, video = vi)
        print("pickle file of the reward per episode was saved! ('rewards_stat')")
    elif y == 3 or y == 4:
        # Watch agent | record
        env = gym.make("BreakoutDeterministic-v4") # <- 
        conv_layer_sizes =  [(32,8,4), (64, 4,2), (64,3,1)] # in the paper it is [(16, 8,4), (32, 4,2)] +relu for both
        hidden_layer_sizes =[[512, tf.nn.relu]] # [[512, tf.nn.relu]] # in the paper [256, relu]
    
        model = DQN(num_channels = 4, image_size = 84, K = K, conv_layer_sizes= conv_layer_sizes,
                    hidden_layer_sizes = hidden_layer_sizes,scope = 'model')
        
        model.load('paramsModel_modified_DDQN_Deterministic_v4')
        image_transformer = ImgaeTransformer()
        
        if y ==3:
            # Watch agent 
            watch_agent(env, model, image_transformer = image_transformer)
            
        
        else:
            # recored Agent 
            print('Recording...please wait')
            record_agent(env, model,image_transformer = image_transformer, videoNum = 'Single_run')
        
        
    else:
        print("Please run again and enter a valid number!")
        
    
    print("Done!!")


            
            
            
    
        
        
        
        
        
        
        
        