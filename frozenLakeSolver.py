import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
'''from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint'''
import tensorflow as tf
import matplotlib.pyplot as plt

class frozenLakeAI:
	def __init__(self,render=False):
		self.env = gym.make('FrozenLake-v0')
		self.env.reset()
		self.render = render
		self.neural_network_model(16)

	def neural_network_model(self,input_size,output_size=4):
		tf.reset_default_graph()

		#These lines establish the feed-forward part of the network used to choose actions
		# input
		self.inputs1 = tf.placeholder(shape=[1,input_size],dtype=tf.float32)

		# output of network
		#output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, out])),'biases':tf.Variable(tf.random_normal([output_size]))}
		self.output_layer = tf.Variable(tf.random_uniform([input_size,output_size],0,0.01))
		self.Qout = tf.matmul(self.inputs1,self.output_layer)

		# prediction from network
		self.predict = tf.argmax(self.Qout,1)

		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.nextQ = tf.placeholder(shape=[1,output_size],dtype=tf.float32)
		self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
		self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
		self.updateModel = self.trainer.minimize(self.loss)

	def fit(self,num_episodes=2000,learningRateDecay=0.5,name=None):
		init = tf.global_variables_initializer()
		'''callbacks = []
		callbacks.append(EarlyStopping(monitor='loss',patience=10,mode='auto',min_delta=0.01,verbose=0))
		callbacks.append(ReduceLROnPlateau(monitor='loss', factor=learningRateDecay, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0))
		if name:
			checkpoint_path = 'checkpoints/weights.{}.hdf5'.format(name)
			callbacks.append(ModelCheckpoint(checkpoint_path,save_weights_only=True,save_best_only=True))'''

		# set learning parameters 
		gamma = 0.99
		eps = 0.1

		# create lists to contain scores and steps per episode
		scores = []
		steps = []
		with tf.Session() as sess:
			sess.run(init)
			for i in range(num_episodes):
				# reset enviroment and get first observation
				observation = self.env.reset()
				score = 0
				done = False
				step = 0

				# the Q-network
				while step < 99:
					step+=1

					# choose an action by greedily (with probability episilon of random action) from the Q-network
					action, target = sess.run([self.predict,self.Qout],feed_dict={self.inputs1:np.identity(16)[observation:observation+1]})
					if np.random.rand(1) < eps:
						action[0] = self.env.action_space.sample()

					# get new observation and reward from enviroment
					new_observation, reward, done, _ = self.env.step(action[0])

					# obtain the Q' values by feeding the new observation through our network
					target1 = sess.run(self.Qout,feed_dict={self.inputs1:np.identity(16)[new_observation:new_observation+1]})

					# obtain max Q' and set our target value for chosen action
					maxTarget1 = np.max(target1)
					targetQ = target
					targetQ[0,action[0]] = reward + gamma*maxTarget1

					# train our network using target and predicet Q values
					_,W1 = sess.run([self.updateModel,self.output_layer],feed_dict={self.inputs1:np.identity(16)[observation:observation+1],self.nextQ:targetQ})

					score += reward
					observation = new_observation

					if done == True:
						# reduce the chance of taking a random action as we train the model
						eps = 1./((i/50) + 10)
						break

				scores.append(score)
				steps.append(step)
		print("Percent of succesful episodes: " + str(sum(scores)/num_episodes) + "%")

		plt.plot(scores)
		plt.plot(steps)

if __name__ == '__main__':
	AI = frozenLakeAI(render=False)
	AI.fit(num_episodes=2000)
