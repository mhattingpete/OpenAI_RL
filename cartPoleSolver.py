import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class cartPoleAI:
	def __init__(self,render=False):
		self.env = gym.make('CartPole-v0')
		self.env.reset()
		self.goal_steps = 500
		self.score_requirement = 50
		self.initial_games = 10000
		self.render = render
		training_data = self.initial_population()
		self.X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]))
		self.y = [i[1] for i in training_data]
		self.model = []
		self.neural_network_model(input_size=len(self.X[0]))

	def initial_population(self):
		# [OBS, MOVES]
		training_data = []
		# all scores:
		scores = []
		# just the scores that met our threshold:
		accepted_scores = []
		# iterate through however many games we want:
		for _ in range(self.initial_games):
			score = 0
			# moves specifically from this environment:
			game_memory = []
			# previous observation that we saw
			prev_observation = []
			# for each frame in 200
			for _ in range(self.goal_steps):
				#self.env.render()
				# choose random action (0 or 1)
				action = random.randrange(0,2)
				# do it!
				observation, reward, done, info = self.env.step(action)
				# notice that the observation is returned FROM the action
				# so we'll store the previous observation here, pairing
				# the prev observation to the action we'll take.
				if len(prev_observation) > 0 :
					game_memory.append([prev_observation, action])
				prev_observation = observation
				score+= reward
				if done: break

			# IF our score is higher than our threshold, we'd like to save
			# every move we made
			# NOTE the reinforcement methodology here. 
			# all we're doing is reinforcing the score, we're not trying 
			# to influence the machine in any way as to HOW that score is 
			# reached.
			if score >= self.score_requirement:
				accepted_scores.append(score)
				for data in game_memory:
					# convert to one-hot (this is the output layer for our neural network)
					if data[1] == 1:
						output = [0,1]
					elif data[1] == 0:
						output = [1,0]

					# saving our training data
					training_data.append([data[0], output])

			# reset env to play again
			self.env.reset()
			# save overall scores
			scores.append(score)

		# just in case you wanted to reference later
		#training_data_save = np.array(training_data)
		#np.save('saved.npy',training_data_save)

		# some stats here, to further illustrate the neural network magic!
		print('Average accepted score:',mean(accepted_scores))
		print('Median score for accepted scores:',median(accepted_scores))
		print(Counter(accepted_scores))

		return training_data

	def neural_network_model(self,input_size,reg_factor=0.03,dropout_factor=0.2):
		self.optimizer = Adam()

		i = Input(shape=[input_size])

		x = Dense(128,activation='relu',W_regularizer=l2(reg_factor))(i)
		x = Dropout(dropout_factor)(x)

		x = Dense(64,activation='relu',W_regularizer=l2(reg_factor))(i)
		x = Dropout(dropout_factor)(x)

		o = Dense(2,activation='softmax')(x)

		self.model = Model(input=i,output=o)
		self.model.compile(optimizer=self.optimizer,loss='categorical_crossentropy',metrics=['acc'])

	def fit(self,epochs=5,learningRateDecay=0.5,name=None):
		callbacks = []
		callbacks.append(EarlyStopping(monitor='loss',patience=10,mode='auto',min_delta=0.01,verbose=2))
		callbacks.append(ReduceLROnPlateau(monitor='loss', factor=learningRateDecay, patience=2, verbose=2, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0))
		if name:
			checkpoint_path = 'checkpoints/weights.{}.hdf5'.format(name)
			callbacks.append(ModelCheckpoint(checkpoint_path,save_weights_only=True,save_best_only=True))

		self.model.fit(self.X,self.y, nb_epoch=epochs, batch_size=32,verbose=2,callbacks=callbacks)

	def evaulate_model(self):
		scores = []
		choices = []
		for each_game in range(100):
			score = 0
			game_memory = []
			prev_obs = []
			self.env.reset()
			for _ in range(self.goal_steps):
				if self.render:
					self.env.render()

				if len(prev_obs) == 0:
					action = random.randrange(0,2)
				else:
					action = np.argmax(self.model.predict(prev_obs.reshape(-1,len(prev_obs)))[0])

				choices.append(action)

				new_observation, reward, done, info = self.env.step(action)
				prev_obs = new_observation
				game_memory.append([new_observation,action])
				score += reward
				if done: break

			scores.append(score)

		print('Average score:',sum(scores)/len(scores))
		print('choice 1:{} choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
		print(self.score_requirement)

if __name__ == '__main__':
	AI = cartPoleAI(render=False)
	AI.fit(epochs=20)
	AI.evaulate_model()
