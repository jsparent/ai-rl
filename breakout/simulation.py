import numpy as np
import sys
import plotting as plotting
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import pylab
import matplotlib.gridspec as gridspec
import time

class Experiment(object):
    def __init__(self, env, agent, EPISODES=1000, training=True, episode_max_length=None, mean_episodes=10, stop_criterion=100):
        self.start_time = time.time()
        self.env = env
        self.agent = agent
        self.EPISODES = EPISODES
        self.training = training
        self.episode_max_length = episode_max_length
        self.mean_episodes = mean_episodes
        self.stop_criterion = stop_criterion
        self.high_score = 0
        
        self.episode_length = np.array([0])
        self.episode_reward = np.array([0])
                
        # Partie graphique
        self.fig = pylab.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(2, 2)
        self.ax = pylab.subplot(gs[:, 0])
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        
        if hasattr(self.env, '_cliff'): # Hardcode to nicely display grid for cliffwalkingenv
            self.ax.xaxis.set_visible(True)
            self.ax.yaxis.set_visible(True)
            self.ax.set_xticks(np.arange(-.5, 12, 1), minor=True);
            self.ax.set_yticks(np.arange(-.5, 4, 1), minor=True);
            self.ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
            
        if hasattr(self.env, 'winds'): # Hardcode to nicely display grid for windygridworldenv
            self.ax.xaxis.set_visible(True)
            self.ax.yaxis.set_visible(True)
            self.ax.set_xticks(np.arange(-.5, 10, 1), minor=True);
            self.ax.set_yticks(np.arange(-.5, 7, 1), minor=True);
            self.ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
        
        self.ax1 = pylab.subplot(gs[0, 1])
        self.ax1.yaxis.set_label_position("right")
        self.ax1.set_ylabel('Length')
        
        self.ax1.set_xlim(0, max(10, len(self.episode_length)+1))
        self.ax1.set_ylim(0, 51)
        
        self.ax2 = pylab.subplot(gs[1, 1])
        self.ax2.set_xlabel('Episode')
        self.ax2.yaxis.set_label_position("right")
        self.ax2.set_ylabel('Reward')
        self.ax2.set_xlim(0, max(10, len(self.episode_reward)+1))
        self.ax2.set_ylim(0, 2)
        
        self.line, = self.ax1.plot(range(len(self.episode_length)),self.episode_length)
        self.line2, = self.ax2.plot(range(len(self.episode_reward)),self.episode_reward)
        
    def update_display_step(self, step = None):
        if not hasattr(self, 'imgplot'):
            self.imgplot = self.ax.imshow(self.env.render(mode='rgb_array'), interpolation='none', cmap='viridis')
        else:
            self.imgplot.set_data(self.env.render(mode='rgb_array'))
        
        self.fig.canvas.draw()
        #if not self.training:
        #   time.sleep(0.05) #5/100 de seconde
        
             
    def update_display_episode(self):  
        self.line.set_data(range(len(self.episode_length)),self.episode_length)
        self.ax1.set_xlim(0, max(10, len(self.episode_length)+1))
        self.ax1.set_ylim(0, max(self.episode_length)+1)
        
        self.line2.set_data(range(len(self.episode_reward)),self.episode_reward)
        self.ax2.set_xlim(0, max(10, len(self.episode_reward)+1))
        self.ax2.set_ylim(min(self.episode_reward)-1, max(self.episode_reward)+1)
        
        self.fig.canvas.draw()     
        
    def start_run(self):
        self.start_time = time.time()
        start_time_display = time.strftime("%Y-%m-%d %H:%M:%S")
        self.agent.log("--------------------------------------------------------")
        self.agent.log(f"Nouvelle séquence de {self.EPISODES} épisodes, débutée le {start_time_display}...")
        self.agent.log("--------------------------------------------------------")
        self.high_score = 0
        if not self.training:
            self.agent.log("> Mode REPLAY")
            
    def end_run(self):
        # Enregistrer le modèle à la fin
        if self.training:
            self.agent.saveModel()
        
        running_total_seconds = round(time.time() - self.start_time, 0)
        running_total_minutes = round(running_total_seconds / 60, 1)
        self.agent.log(f"- Temps de traitement: {running_total_minutes} minutes")
        self.agent.log("--------------------------------------------------------")
        self.agent.log("", flushBuffer=True)
        
    def run_qlearning(self, interactive = False, display_frequency=1, save_model_each_n_episodes = 50, time_penalty=0, life_penalty=0):
        self.start_run()
        
        BYTE_VIE = 57

        # repeat for each episode
        for episode_number in range(self.EPISODES):
                
            # initialize state            
            state = self.env.reset()
            
            done = False # used to indicate terminal state
            R = 0 # used to display accumulated rewards for an episode
            t = 0 # used to display accumulated steps for an episode i.e episode length
            
            # repeat for each step of episode, until state is terminal
            while not done:
                
                t += 1 # increase step counter - for display
                
                # choose action from state using policy derived from Q
                action = self.agent.act(state)
                
                # take action, observe reward and next state
                next_state, reward, done, _ = self.env.step(action)

                # Pénaliser le fait de ne rien faire pour éviter des épisodes qui s'étirent
                learning_reward = reward
                
                # Après discussions avec Mikael, ne pas pénaliser l'inaction ou les vies
                if learning_reward == 0:
                    vies_actuelles = state[BYTE_VIE]
                    vies_apres = next_state[BYTE_VIE]
                    if vies_apres < vies_actuelles:
                        #print("vie perdue")
                        learning_reward = -life_penalty # Pénaliser fortement la perte de vie
                    else:
                        learning_reward = -time_penalty # Pénaliser légèrement l'inaction
                    
                # agent learn (Q-Learning update)
                if self.training:
                    self.agent.learn(state, action, learning_reward, next_state, done)
                
                # state <- next state
                state = next_state
                
                R += reward # accumulate reward - for display
                
                # if interactive display, show update for each step 
                #if self.training and interactive:
                self.update_display_step(t)
                
                 # If cancel requested, exit
                if self.agent.isCancelRequested():
                    self.agent.log("*** Arrêt demandé détecté ***", flushBuffer=True)
                    break;

                    
            self.episode_length = np.append(self.episode_length,t) # keep episode length - for display
            self.episode_reward = np.append(self.episode_reward,R) # keep episode reward - for display 


            if R > 0:
                self.agent.log(f"Épisode {episode_number+1}/{self.EPISODES}: R={R}, Steps={t}", doPrint=not interactive)
            
            # Update image of highest score only
            if interactive:
                if R >= self.high_score:
                    self.update_display_step()
                self.update_display_episode()
                
            if R > self.high_score:
                self.high_score = R
                self.agent.log(f"\tNouveau meilleur score à: {self.high_score}, épisode #{episode_number + 1}", flushBuffer=True)
                
            # Sauvegarde du modèle tous les n épisodes
            if self.training and save_model_each_n_episodes != None and (episode_number + 1) % save_model_each_n_episodes == 0:
                self.agent.saveModel()
            
            # if interactive display, show update for the episode
            if not self.training and interactive:
                self.update_display_episode()
                
            # If cancel requested, exit
            if self.agent.isCancelRequested():
                self.agent.log("*** Arrêt demandé par l'usager ***", flushBuffer=True)
                break;
                
            
        # if interactive display, show graph at the end
        if interactive:
            self.update_display_episode()
                    
        else:
            self.fig.clf()
            stats = plotting.EpisodeStats(
                episode_lengths=self.episode_length,
                episode_rewards=self.episode_reward,
                episode_running_variance=np.zeros(self.EPISODES))
            plotting.plot_episode_stats(stats, display_frequency)
        
        self.agent.log("")
        self.agent.log(f"Fin des épisodes")
        self.agent.log(f"Meilleur score obtenu: {self.high_score}")
        self.agent.log(f"Durée moyenne: {round(np.average(self.episode_length), 1)} actions")
        self.agent.log(f"Score moyen: {round(np.average(self.episode_reward), 2)} points")
        self.agent.log("", flushBuffer=True)
        
        self.end_run()
        
        
    def run_actorcritic(self):
        self.start_run()
        
        # Tableaux utiles pour l'affichage
        scores, mean, episodes, lengths = [], [], [], []
        
        plt.ion()
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 8
        fig_size[1] = 7
        
        fig1, (ax1_1, ax1_2, ax1_3) = plt.subplots(3, sharex=True)
        fig1.canvas.draw()

        for i in range(self.EPISODES):
            done = False
            score = 0
            state = self.env.reset()

            counter = 0
            while not done:
                counter +=1

                # Afficher l'environnement
                if self.agent.render:
                    self.env.render()

                # Obtient l'action pour l'état courant
                action = self.agent.act(state)

                # Effectue l'action
                next_state, reward, done, _ = self.env.step(action)

                self.agent.learn(state, action, reward, next_state, done )

                # Mise à jour de l'état
                state = next_state

                # Accumulation des récompenses
                score += reward

                # Arrête l'épisode après 'episode_max_length' instants
                if self.episode_max_length != None and counter >= self.episode_max_length:
                    done = True
           
            
            if score > self.high_score:
                self.high_score = score
                self.agent.log(f"\tNouveau meilleur score à: {self.high_score}, épisode #{i + 1}", flushBuffer=True)
           
            # Arrête l'entraînement lorsque la moyenne des récompense sur 'mean_episodes' épisodes est supérieure à 
            if np.mean(scores[-self.mean_episodes:]) > self.stop_criterion:
                break

            # Sauvegarde du modèle (poids) tous les 25 épisodes
            if self.training and (i + 1) % 25 == 0:
                self.agent.saveModel()
                self.agent.log(f"\tÉpisode {i + 1}", flushBuffer=True)
                
            # Affichage des récompenses obtenues
            if self.training == True:
                scores.append(score)
                mean.append(np.mean(scores[-self.mean_episodes:]))
                episodes.append(i)
                lengths.append(counter)
                
                ax1_1.clear()
                ax1_1.plot(episodes, scores, 'b', label='gains')
    
                ax1_2.clear()
                ax1_2.plot(episodes, mean, 'r', label='Moyenne des gains')
                
                ax1_3.clear()
                ax1_3.plot(episodes, lengths, 'g', label='Durée')
    
                fig1.canvas.draw()
                
                #ax1_1.set_xlabel("Épisodes")
                ax1_1.set_ylabel("Gains")
                #ax1_2.set_xlabel("Épisodes")
                ax1_2.set_ylabel("Moyenne des gains")
                ax1_3.set_xlabel("Épisodes")
                ax1_3.set_ylabel("Durée")
                #plt.legend(loc='upper left')
             
            # If cancel requested, exit
            if self.agent.isCancelRequested():
                self.agent.log("*** Arrêt demandé par l'usager ***", flushBuffer=True)
                break;
                
        self.agent.log("")
        self.agent.log(f"Fin des épisodes")
        self.agent.log(f"- Meilleur score obtenu: {self.high_score}")
        self.agent.log(f"- Durée moyenne: {round(np.average(lengths), 1)} actions")
        self.agent.log(f"- Score moyen: {round(np.average(scores), 2)} points")
        self.agent.log("", flushBuffer=True)
        
        self.end_run()