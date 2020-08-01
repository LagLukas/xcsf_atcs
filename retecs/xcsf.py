import unittest
import copy
from classifier import XCSF_Classifier
from matching import XCSFMatching
from action_selection import XCSFActionSelection
from reinforcement import Reinforcement
from genetic_algorithm import CIGeneticAlgorithm
import random
import pickle


class XCSF:
    '''
    dirty code of a stressed programmer, will clean it up later :)
    '''
    GAMMA = 0.71

    def __init__(self, max_population_size, possible_actions=None, histlen=42):
        print("init XCSF")
        self.name = "XCSF"
        self.action_size = len(possible_actions)
        self.max_population_size = max_population_size
        if possible_actions is None:
            self.possible_actions = [-10, 10]
        else:
            self.possible_actions = possible_actions
        self.population = []
        self.time_stamp = 1
        self.action_history = []
        self.old_action_history = []
        self.reinforce = Reinforcement()
        self.ga = CIGeneticAlgorithm(possible_actions)
        #################################
        self.single_testcases = True
        self.histlen = histlen
        #################################
        # stuff for batch update
        self.max_prediction_sum = 0
        self.rewards = None
        self.p_explore = 0.25
        self.train_mode = True
        self.cycle = 0

    def get_action(self, state):
        '''
        :param state: State in Retects. In the XCS world = situation.

        :return : a action
        '''
        theta_mna = 45 # len(self.possible_actions)
        matcher = XCSFMatching(theta_mna, self.possible_actions)
        match_set = matcher.get_match_set(self.population, state, self.time_stamp)
        self.p_explore = (self.p_explore - 0.1) * 0.99 + 0.1
        action_selector = XCSFActionSelection(self.possible_actions, self.p_explore)
        best_possible_action = action_selector.get_best_action(match_set, state)
        chosen_action = action_selector.get_action(best_possible_action, self.train_mode)
        # calculate system prediction
        fitness_sum = sum(list(map(lambda x: x.fitness, self.population)))
        system_prediction = sum(list(map(lambda x: x.fitness * x.get_target(state, chosen_action), self.population)))
        if fitness_sum > 0:
            system_prediction = system_prediction / fitness_sum
        max_val = system_prediction # on policy
        action_set = match_set
        # action_set = action_selector.get_action_set(match_set, action)
        self.max_prediction_sum += max_val
        self.action_history.append((state, action_set, state, chosen_action))
        return system_prediction
        # return chosen_action

    def reward(self, new_rewards):
        try:
            x = float(new_rewards)
            new_rewards = [x] * len(self.action_history)
        except Exception as _:
            if len(new_rewards) < len(self.action_history):
                raise Exception('Too few rewards')
        old_rewards = self.rewards
        self.rewards = new_rewards
        old_rewards = self.rewards
        if old_rewards is not None:
            avg_max_pred = self.max_prediction_sum / len(self.action_history)
            for i in range(0, len(old_rewards)):
                discounted_reward = old_rewards[i] #+ XCSF.GAMMA * avg_max_pred
                old_sigma, old_action_set, old_sigma, old_action = self.action_history[i] # self.old_action_history[i]
                self.reinforce.reinforce_xcsf(old_action_set, discounted_reward, old_sigma, old_action)
                self.ga.perform_iteration(old_action_set, old_sigma, self.population, self.time_stamp)
                self.time_stamp += 1
        self.max_prediction_sum = 0
        self.old_action_history = self.action_history
        self.action_history = []
        self.delete_from_population()
        print("XCSF: finished cycle " + str(self.cycle))
        self.cycle += 1

    def delete_from_population(self):
        '''
        Deletes as many classifiers as necessary until the population size is within the
        defined bounds.
        '''
        total_numerosity = sum(list(map(lambda x: x.numerosity, self.population)))
        while len(self.population) > self.max_population_size:
            total_fitness = sum(list(map(lambda x: x.fitness, self.population)))
            avg_fitness = total_fitness / total_numerosity
            vote_sum = sum(list(map(lambda x: x.deletion_vote(avg_fitness), self.population)))
            choice_point = random.random() * vote_sum
            vote_sum = 0
            for classifier in self.population:
                vote_sum += classifier.deletion_vote(avg_fitness)
                if vote_sum > choice_point:
                    if classifier.numerosity > 1:
                        classifier.numerosity = classifier.numerosity - 1
                    else:
                        self.population.remove(classifier)

    def save(self, filename):
        """ Stores agent as pickled file """
        pickle.dump(self, open(filename + '.p', 'wb'), 2)

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename + '.p', 'rb'))
