import time
from datetime import datetime

import random
import os.path

import gym
from gym.wrappers import Monitor

import tensorflow as tf
from keras import backend as K
from keras.models import load_model

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize


OBSERVE_STEP_NUM = 50000
NUM_EPOCHS = 100000
ATARI_SHAPE = (84, 84, 4) 
ACTION_SIZE = 3

render_game = True
record_game = True
network_path = './train_breakout/breakout_model_20200125073327.h5'
log_path = './log_test.txt'
video_path = './video/'


def show_video():
  mp4list = glob.glob(video_path+'*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
      
def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env
  
  
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe
  
  
def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss
  
    
def get_action(history, epsilon, step, model):
    if np.random.rand() <= epsilon or step <= OBSERVE_STEP_NUM:
        return random.randrange(ACTION_SIZE)
    else:
        q_value = model.predict([history, np.ones(ACTION_SIZE).reshape(1, ACTION_SIZE)])
        return np.argmax(q_value[0])


def test():
    if record_game:
        env =  wrap_env(gym.make('BreakoutDeterministic-v4'))
    else:
        env =  gym.make('BreakoutDeterministic-v4')

    episode_number = 0
    epsilon = 0.001
    global_step = OBSERVE_STEP_NUM+1
    model = load_model(network_path, custom_objects={'huber_loss': huber_loss}) 
    
    file = open(log_path, 'w+')
    file.write('episode,score\n')

    while episode_number < NUM_EPOCHS:

        done = False
        dead = False
        score, start_life = 0, 5
        observe = env.reset()

        observe, _, _, _ = env.step(1)
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            env.render()
            time.sleep(0.01)

            action = get_action(history, epsilon, global_step, model)
            real_action = action + 1

            observe, reward, done, info = env.step(real_action)
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1.)
            score += reward

            if dead:
                dead = False
            else:
                history = next_history

            global_step += 1

            if done:
                file.write(str(episode_number)+','+str(score)+'\n')
                episode_number += 1
                print('episode: {}, score: {}'.format(episode_number, score))
    file.close()
    show_video()


def main(argv=None):
    test()

if __name__ == '__main__':
    main()

