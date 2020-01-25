import time
import random
import os.path
from datetime import datetime

import gym
from gym.wrappers import Monitor

import numpy as np
import tensorflow as tf

from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize

from keras import layers
from keras import backend as K
from keras.optimizers import RMSprop
from keras.models import Model, load_model, clone_model


# Training details
training_dir = 'trained_network'
render_game = False
NUM_EPOCHS = 100000
OBSERVE_STEP_NUM = 50000
EPSILON_STEP_NUM = 1000000
REFRESH_TARGET_MODEL_NUM = 10000
REPLAY_MEMORY = 400000
NO_OP_STEPS = 30
REGULARIZER_SCALE = 0.01
BATCH_SIZE = 32
LEARNING_RATE = 0.00025
INIT_EPSILON = 1.0
FINAL_EPSILON = 0.1
GAMMA = 0.99

ATARI_SHAPE = (84, 84, 4) 
ACTION_SIZE = 3

# Resume model training
resume = True
restore_path = './trained_network/breakout_model_20200125073327.h5'


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


def atari_model():
    frames_input = layers.Input(ATARI_SHAPE, name='frames')
    actions_input = layers.Input((ACTION_SIZE,), name='action_mask')
    normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

    conv_1 = layers.convolutional.Conv2D(
        32, (8, 8), strides=(4, 4), activation='relu'
    )(normalized)
    conv_2 = layers.convolutional.Conv2D(
        64, (4, 4), strides=(2, 2), activation='relu'
    )(conv_1)
    conv_3 = layers.convolutional.Conv2D(
        64, (3, 3), subsample=(1, 1), activation='relu'
    )(conv_2)

    conv_flattened = layers.core.Flatten()(conv_3)
    hidden = layers.Dense(512, activation='relu')(conv_flattened)
    output = layers.Dense(ACTION_SIZE)(hidden)
    
    filtered_output = layers.Multiply(name='QValue')([output, actions_input])

    model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    model.summary()
    optimizer = RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss=huber_loss)
    return model


def get_action(history, epsilon, step, model):
    if np.random.rand() <= epsilon or step <= OBSERVE_STEP_NUM:
        return random.randrange(ACTION_SIZE)
    else:
        q_value = model.predict([history, np.ones(ACTION_SIZE).reshape(1, ACTION_SIZE)])
        return np.argmax(q_value[0])


def store_memory(memory, history, action, reward, next_history, dead):
    memory.append((history, action, reward, next_history, dead))


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def train_memory_batch(memory, model, log_dir):
    mini_batch = random.sample(memory, BATCH_SIZE)
    history = np.zeros((BATCH_SIZE, ATARI_SHAPE[0],
                        ATARI_SHAPE[1], ATARI_SHAPE[2]))
    next_history = np.zeros((BATCH_SIZE, ATARI_SHAPE[0],
                             ATARI_SHAPE[1], ATARI_SHAPE[2]))
    target = np.zeros((BATCH_SIZE,))
    action, reward, dead = [], [], []

    for idx, val in enumerate(mini_batch):
        history[idx] = val[0]
        next_history[idx] = val[3]
        action.append(val[1])
        reward.append(val[2])
        dead.append(val[4])

    actions_mask = np.ones((BATCH_SIZE, ACTION_SIZE))
    next_Q_values = model.predict([next_history, actions_mask])

    for i in range(BATCH_SIZE):
        if dead[i]:
            target[i] = -1
        else:
            target[i] = reward[i] + GAMMA * np.amax(next_Q_values[i])

    action_one_hot = get_one_hot(action, ACTION_SIZE)
    target_one_hot = action_one_hot * target[:, None]

    h = model.fit(
        [history, action_one_hot], target_one_hot, epochs=1,
        batch_size=BATCH_SIZE, verbose=0)

    return h.history['loss'][0]
    

def train():
    env = gym.make('BreakoutDeterministic-v4')

    memory = deque(maxlen=REPLAY_MEMORY)
    episode_number = 0
    epsilon = INIT_EPSILON
    epsilon_decay = (INIT_EPSILON - FINAL_EPSILON) / EPSILON_STEP_NUM
    global_step = 0

    if resume:
        model = load_model(restore_path, custom_objects={'huber_loss': huber_loss})
        epsilon = FINAL_EPSILON
    else:
        model = atari_model()

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    model_target = clone_model(model)
    model_target.set_weights(model.get_weights())
    
    log_path = './log_training/log.txt'
    file = open(log_path, 'w+')
    file.write('episode,score,global_step,avg_loss,step\n')

    while episode_number < NUM_EPOCHS:
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        loss = 0.0
        observe = env.reset()

        for _ in range(random.randint(1, NO_OP_STEPS)):
            observe, _, _, _ = env.step(1)
            
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))
        
        while not done:
            if render_game:
                env.render()
                time.sleep(0.01)

            action = get_action(history, epsilon, global_step, model_target)
            real_action = action + 1

            if epsilon > FINAL_EPSILON and global_step > OBSERVE_STEP_NUM:
                epsilon -= epsilon_decay

            observe, reward, done, info = env.step(real_action)

            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']
                
            store_memory(memory, history, action, reward, next_history, dead)  #

            if global_step > OBSERVE_STEP_NUM:
                loss = loss + train_memory_batch(memory, model, log_dir)
                if global_step % REFRESH_TARGET_MODEL_NUM == 0:  
                    model_target.set_weights(model.get_weights())

            score += reward
            if dead:
                dead = False
            else:
                history = next_history

            global_step += 1
            step += 1

            if done:
                file.write(str(episode_number)+','+str(score)+','+str(global_step)+','+str(loss / float(step))+','+str(step)+'\n')
                if global_step <= OBSERVE_STEP_NUM:
                    state = "observe"
                elif OBSERVE_STEP_NUM < global_step <= OBSERVE_STEP_NUM + EPSILON_STEP_NUM:
                    state = "explore"
                else:
                    state = "train"
                print('state: {}, episode: {}, score: {}, global_step: {}, avg loss: {}, step: {}, memory length: {}'
                      .format(state, episode_number, score, global_step, loss / float(step), step, len(memory)))

                if episode_number % 100 == 0 or (episode_number + 1) == NUM_EPOCHS:
                    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                    file_name = "breakout_model_{}.h5".format(now)
                    model_path = os.path.join(training_dir, file_name)
                    model.save(model_path)

                episode_number += 1
                  
    file.close()


def main(argv=None):
    train()


if __name__ == '__main__':
    train()
