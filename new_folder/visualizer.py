import pygame
import win32gui
import win32con

# 定义常量
WIDTH, HEIGHT = 400, 400
BLOCK_SIZE = WIDTH // 4
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# 绘制单个方块
def draw_block(screen, x, y, num):
    pygame.draw.rect(screen, GRAY, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    font = pygame.font.Font(None, 36)
    text = font.render(str(num), True, BLACK)
    text_rect = text.get_rect(center=(x * BLOCK_SIZE + BLOCK_SIZE // 2, y * BLOCK_SIZE + BLOCK_SIZE // 2))
    screen.blit(text, text_rect)

# 绘制整个拼图
def draw_puzzle(screen, puzzle):
    screen.fill(WHITE)
    for y in range(4):
        for x in range(4):
            num = puzzle[y][x]
            if num != 0:
                draw_block(screen, x, y, num)
    pygame.display.flip()

def play_animation(solution, delay=1000):
    try:
        # 初始化 Pygame
        pygame.init()
        # 创建窗口
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Puzzle Animation")

        # 获取窗口句柄并将窗口置顶
        hwnd = win32gui.FindWindow(None, "Puzzle Animation")
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

        clock = pygame.time.Clock()
        for state in solution:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            draw_puzzle(screen, state)
            pygame.time.delay(delay)
            clock.tick(60)
    except pygame.error as e:
        print(f"Pygame error: {e}")
    finally:
        # 确保在任何情况下都能退出 Pygame
        pygame.quit()

# 在程序结束时退出 Pygame
def quit_pygame():
    pygame.quit()