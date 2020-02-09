from operator import add
from random import randint


import numpy as np
import pygame
from gym import spaces


class SnakeEnv(object):

    def __init__(self, game_width=440, game_height=440):
        pygame.init()
        pygame.display.set_caption('Snakedqn')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("img/background.png")

        self.player = Player(self)
        self.food = Food()
        self.score = 0
        self.record = 0

        self.done = False
        self.reward = 0
        self.state_dim = 11  # 十一个观测值
        self.action_dim = 3  # 三个动作
        self.state = []

        self.food_image = pygame.image.load('img/food2.png')
        self.play_image = pygame.image.load('img/snakeBody.png')

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self.player.do_move(action, self.player.x, self.player.y, self, self.food)
        self.state = self.get_state(self.player, self.food)
        # set treward for the new state
        self.set_reward()

        return self.state, self.reward, self.done, {}

    def reset(self):
        self.done = False
        self.reward = 0
        self.score = 0
        self.player.__init__(self)
        self.food.__init__()
        self.state = self.get_state(self.player, self.food)
        return self.state

    def render(self, speed=0):
        self.gameDisplay.fill((255, 255, 255))
        self.display_ui(self.score, self.record)
        self.player.display_player(self.player.position[-1][0], self.player.position[-1][1], self.player.length, self)
        self.food.display_food(self)
        pygame.display.update()
        pygame.time.wait(speed)
        return None

    def display_ui(self, score, record):
        myfont = pygame.font.SysFont('Segoe UI', 20)
        myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
        text_score = myfont.render('SCORE: ', True, (0, 0, 0))
        text_score_number = myfont.render(str(score), True, (0, 0, 0))
        text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
        text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
        self.gameDisplay.blit(text_score, (45, 440))
        self.gameDisplay.blit(text_score_number, (120, 440))
        self.gameDisplay.blit(text_highest, (190, 440))
        self.gameDisplay.blit(text_highest_number, (350, 440))
        self.gameDisplay.blit(self.bg, (10, 10))

    def close(self):
        return None

    def get_state(self, player, food):

        state = [
            (player.x_change == 20 and player.y_change == 0 and (
                    (list(map(add, player.position[-1], [20, 0])) in player.position) or
                    player.position[-1][0] + 20 >= (self.game_width - 20))) or (
                    player.x_change == -20 and player.y_change == 0 and (
                    (list(map(add, player.position[-1], [-20, 0])) in player.position) or
                    player.position[-1][0] - 20 < 20)) or (player.x_change == 0 and player.y_change == -20 and (
                    (list(map(add, player.position[-1], [0, -20])) in player.position) or
                    player.position[-1][-1] - 20 < 20)) or (player.x_change == 0 and player.y_change == 20 and (
                    (list(map(add, player.position[-1], [0, 20])) in player.position) or
                    player.position[-1][-1] + 20 >= (self.game_height - 20))),  # danger straight

            (player.x_change == 0 and player.y_change == -20 and (
                    (list(map(add, player.position[-1], [20, 0])) in player.position) or
                    player.position[-1][0] + 20 > (self.game_width - 20))) or (
                    player.x_change == 0 and player.y_change == 20 and ((list(map(add, player.position[-1],
                                                                                  [-20, 0])) in player.position) or
                                                                        player.position[-1][0] - 20 < 20)) or (
                    player.x_change == -20 and player.y_change == 0 and ((list(map(
                add, player.position[-1], [0, -20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (
                    player.x_change == 20 and player.y_change == 0 and (
                    (list(map(add, player.position[-1], [0, 20])) in player.position) or player.position[-1][
                -1] + 20 >= (self.game_height - 20))),  # danger right

            (player.x_change == 0 and player.y_change == 20 and (
                    (list(map(add, player.position[-1], [20, 0])) in player.position) or
                    player.position[-1][0] + 20 > (self.game_width - 20))) or (
                    player.x_change == 0 and player.y_change == -20 and ((list(map(
                add, player.position[-1], [-20, 0])) in player.position) or player.position[-1][0] - 20 < 20)) or (
                    player.x_change == 20 and player.y_change == 0 and (
                    (list(map(add, player.position[-1], [0, -20])) in player.position) or player.position[-1][
                -1] - 20 < 20)) or (
                    player.x_change == -20 and player.y_change == 0 and (
                    (list(map(add, player.position[-1], [0, 20])) in player.position) or
                    player.position[-1][-1] + 20 >= (self.game_height - 20))),  # danger left

            player.x_change == -20,  # move left
            player.x_change == 20,  # move right
            player.y_change == -20,  # move up
            player.y_change == 20,  # move down
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y  # food down
        ]

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0
        return np.asarray(state)

    def set_reward(self):
        if self.done:
            self.reward -= 10
            return self.reward
        if self.player.eaten:
            self.reward += 10
        else:
            self.reward -= 0.01
        return self.reward

    # 可以加入步数-0.01的reward

    def eat(self, player, food):
        if player.x == food.x_food and player.y == food.y_food:
            food.food_coord(self, player)
            player.eaten = True
            self.score = self.score + 1
            self.record = self.get_record()

    def get_record(self):
        if self.score >= self.record:
            return self.score
        else:
            return self.record


class Player(object):

    def __init__(self, game):
        x = 0.45 * game.game_width
        y = 0.5 * game.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.position.append([self.x-20, self.y])
        self.position.append([self.x, self.y])
        self.length = 2
        self.eaten = False

        self.x_change = 20
        self.y_change = 0

    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.length > 1:
                for i in range(0, self.length - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, x, y, game, food):
        move_array = [self.x_change, self.y_change]

        if move == 0:  # if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif move == 1 and self.y_change == 0:  # if np.array_equal(move, [0, 1, 0])   # right - going horizontal
            move_array = [0, self.x_change]
        elif move == 1 and self.x_change == 0:  # if np.array_equal(move, [0, 1, 0])# right - going vertical
            move_array = [-self.y_change, 0]
        elif move == 2 and self.y_change == 0:  # if np.array_equal(move, [0, 0, 1])# left - going horizontal
            move_array = [0, -self.x_change]
        elif move == 2 and self.x_change == 0:  # if np.array_equal(move, [0, 0, 1])# left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if self.x < 20 or self.x > game.game_width - 40 or self.y < 20 or self.y > game.game_height - 40 or \
                [self.x, self.y] in self.position:
            game.done = True
        game.eat(self, food)
        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.length = self.length + 1
        self.update_position(self.x, self.y)

    def display_player(self, x, y, food, game):
        self.position[-1][0] = x
        self.position[-1][1] = y

        if not game.done:
            for i in range(food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(game.play_image, (x_temp, y_temp))


class Food(object):

    def __init__(self):
        self.x_food = 240
        self.y_food = 200

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.food_coord(game, player)

    def display_food(self, game):
        game.gameDisplay.blit(game.food_image, (self.x_food, self.y_food))
