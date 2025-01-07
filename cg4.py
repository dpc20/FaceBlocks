import pygame
import random
import cv2
import numpy as np
# import signal_create
from yolo import process_frame


# 初始化 Pygame
pygame.font.init()

# GLOBALS VARS
s_width = 1200  # 窗口宽度
s_height = 700  # 窗口高度
play_width = 300  # 游戏区宽度（维持原比例）
play_height = 600  # 游戏区高度（维持原比例）
block_size = 30  # 方块大小

top_left_x = (s_width - play_width) // 4  # 调整游戏区域位置
top_left_y = (s_height - play_height) // 2

# 初始化摄像头
camera = cv2.VideoCapture(0)
camera_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
camera_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))


# SHAPE FORMATS

S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
# index 0 - 6 represent shape


class Piece(object):
    rows = 20  # y
    columns = 10  # x

    def __init__(self, column, row, shape):
        self.x = column
        self.y = row
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0  # number from 0-3


def create_grid(locked_positions={}):
    grid = [[(0,0,0) for x in range(10)] for x in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j,i) in locked_positions:
                c = locked_positions[(j,i)]
                grid[i][j] = c
    return grid


def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(shape, grid):
    accepted_positions = [[(j, i) for j in range(10) if grid[i][j] == (0,0,0)] for i in range(20)]
    accepted_positions = [j for sub in accepted_positions for j in sub]
    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_positions:
            if pos[1] > -1:
                return False

    return True


def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False


def get_shape():
    global shapes, shape_colors

    return Piece(5, 0, random.choice(shapes))


def draw_text_middle(text, size, color, surface):
    font = pygame.font.SysFont('comicsans', size, bold=True)
    label = font.render(text, 1, color)

    surface.blit(label, (top_left_x + play_width/2 - (label.get_width() / 2), top_left_y + play_height/2 - label.get_height()/2))


def draw_grid(surface, row, col):
    sx = top_left_x
    sy = top_left_y
    for i in range(row):
        pygame.draw.line(surface, (128,128,128), (sx, sy+ i*30), (sx + play_width, sy + i * 30))  # horizontal lines
        for j in range(col):
            pygame.draw.line(surface, (128,128,128), (sx + j * 30, sy), (sx + j * 30, sy + play_height))  # vertical lines

def clear_rows(grid, locked):
    inc = 0
    for i in range(len(grid)-1,-1,-1):
        row = grid[i]
        if (0, 0, 0) not in row:
            inc += 1
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j, i)]
                except:
                    continue
    if inc > 0:
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                newKey = (x, y + inc)
                locked[newKey] = locked.pop(key)
    return inc


def draw_next_shape(shape, surface):
    """
    绘制下一个方块的信息
    """
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Next Shape', 1, (255, 255, 255))

    # 调整下一个方块的显示位置
    sx = top_left_x - 185  # 显示在右侧
    sy = top_left_y + 50  # 显示在顶部偏下
    format = shape.shape[shape.rotation % len(shape.shape)]

    # 绘制方块矩阵
    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color, 
                                 (sx + j * block_size, sy + i * block_size, 
                                  block_size, block_size), 0)

    surface.blit(label, (sx, sy - 30))  # 绘制标题


def draw_window(surface, grid, frame, signal, score, next_piece):
    """
    绘制游戏窗口和摄像头内容，同时展示分数和下一个方块
    """
    surface.fill((0, 0, 0))  # 背景为黑色

    # 游戏标题
    font = pygame.font.SysFont('comicsans', 60)
    label = font.render('TETRIS', 1, (255, 255, 255))
    surface.blit(label, (top_left_x + play_width / 2 - (label.get_width() / 2), 30))

    # 绘制网格
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j], 
                             (top_left_x + j * block_size, 
                              top_left_y + i * block_size, 
                              block_size, block_size), 0)

    # 绘制网格线和边框
    draw_grid(surface, 20, 10)
    pygame.draw.rect(surface, (255, 0, 0), 
                     (top_left_x, top_left_y, play_width, play_height), 5)

    # 显示分数
    score_font = pygame.font.SysFont('comicsans', 40)
    score_label = score_font.render(f"Score: {score}", 1, (255, 255, 255))
    surface.blit(score_label, (top_left_x + play_width + 50, top_left_y))

    # 显示下一个方块
    draw_next_shape(next_piece, surface)

    # 在右侧绘制摄像头内容，保持比例
    if frame is not None:
        # 计算适配高度和宽度
        cam_scale = min(600 / camera_height, 600 / camera_width)
        cam_width = int(camera_width * cam_scale)
        cam_height = int(camera_height * cam_scale)

        # 调整摄像头画面大小
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)  # 水平翻转
        frame = cv2.resize(frame, (cam_width, cam_height))  # 调整为目标大小
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        # 计算摄像头显示区域居中位置
        cam_x = s_width - cam_width - 50  # 偏右侧
        cam_y = (s_height - cam_height) // 2
        surface.blit(frame_surface, (cam_x, cam_y))

    # 显示信号
    signal_font = pygame.font.SysFont('comicsans', 40)
    signal_label = signal_font.render(f"Signal: {signal}", 1, (255, 255, 255))
    surface.blit(signal_label, (50, 650))

    pygame.display.update()

# 修改 main 函数
def main():
    global grid
    locked_positions = {}  # (x, y): (255, 0, 0)
    grid = create_grid(locked_positions)

    change_piece = False
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    level_time = 0
    fall_speed = 0.27
    score = 0

    last_signl = 'X'
    ll_signal = 'X'

    last_signal_time = 0  # 信号触发时间记录

    while run:
        current_time = pygame.time.get_ticks()  # 获取当前时间
        ret, frame = camera.read()

        if ret:
            signal, frame = process_frame(frame)  # 处理帧生成信号
        else:
            signal = "NEUTRAL"

        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        level_time += clock.get_rawtime()
        clock.tick()

        if level_time / 1000 > 4:
            level_time = 0
            # 游戏随时间加速
            #if fall_speed > 0.15:
            #    fall_speed -= 0.005

        # 控制信号冷却
        if current_time - last_signal_time > 200:  # 每 100ms 响应一次信号
            if signal == "L" and signal == last_signl :
                current_piece.x -= 1
                if not valid_space(current_piece, grid):
                    current_piece.x += 1
            elif signal == "R" and signal == last_signl :
                current_piece.x += 1
                if not valid_space(current_piece, grid):
                    current_piece.x -= 1
            elif signal == "U" and signal == last_signl and last_signl == ll_signal:
                current_piece.rotation = current_piece.rotation + 1 % len(current_piece.shape)
                if not valid_space(current_piece, grid):
                    current_piece.rotation = current_piece.rotation - 1 % len(current_piece.shape)
            last_signal_time = current_time
        ll_signal = last_signl
        last_signl = signal

        # 方块下落逻辑
        if fall_time / 1000 >= fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not (valid_space(current_piece, grid)) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece.x -= 1
                    if not valid_space(current_piece, grid):
                        current_piece.x += 1

                elif event.key == pygame.K_RIGHT:
                    current_piece.x += 1
                    if not valid_space(current_piece, grid):
                        current_piece.x -= 1

                elif event.key == pygame.K_UP:
                    current_piece.rotation = current_piece.rotation + 1 % len(current_piece.shape)
                    if not valid_space(current_piece, grid):
                        current_piece.rotation = current_piece.rotation - 1 % len(current_piece.shape)

                elif event.key == pygame.K_DOWN:
                    current_piece.y += 1
                    if not valid_space(current_piece, grid):
                        current_piece.y -= 1

        shape_pos = convert_shape_format(current_piece)

        # 添加方块到网格
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                grid[y][x] = current_piece.color

        # 检测是否需要切换方块
        if change_piece:
            score += 5
            for pos in shape_pos:
                locked_positions[(pos[0], pos[1])] = current_piece.color
            current_piece = next_piece
            next_piece = get_shape()
            change_piece = False
            cleared_rows = clear_rows(grid, locked_positions)
            score += cleared_rows * 100

        draw_window(win, grid, frame, signal, score, next_piece)  # 传递下一个方块

        if check_lost(locked_positions):
            run = False

    draw_text_middle("You Lost", 40, (255, 255, 255), win)
    pygame.display.update()
    pygame.time.delay(2000)

def menu(surface):
    """
    游戏菜单界面
    """
    run = True
    while run:
        surface.fill((0, 0, 0))  # 黑色背景

        # 游戏标题
        title_font = pygame.font.SysFont('comicsans', 60)
        title_label = title_font.render("TETRIS", 1, (255, 255, 255))
        surface.blit(title_label, (s_width / 2 - title_label.get_width() / 2, 100))

        # 开始游戏选项
        start_font = pygame.font.SysFont('comicsans', 40)
        start_label = start_font.render("Press SPACE to Start", 1, (255, 255, 255))
        surface.blit(start_label, (s_width / 2 - start_label.get_width() / 2, 300))

        # 退出选项
        quit_label = start_font.render("Press ESC to Quit", 1, (255, 255, 255))
        surface.blit(quit_label, (s_width / 2 - quit_label.get_width() / 2, 400))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # 按下空格开始游戏
                    run = False
                elif event.key == pygame.K_ESCAPE:  # 按下 ESC 退出游戏
                    pygame.quit()
                    quit()

# 设置窗口大小
win = pygame.display.set_mode((s_width, s_height))
pygame.display.set_caption('Tetris with Camera')

# 显示菜单
menu(win)

# 开始游戏
main()

# 释放摄像头资源
camera.release()