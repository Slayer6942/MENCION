import sys, pygame
from pygame.locals import *

WIDTH = 800
HEIGHT = 600


def load_image(filename, transparent=False):
    try:
        image = pygame.image.load(filename)
    except pygame.error, message:
        raise SystemExit, message
    image = image.convert()
    if transparent:
        color = image.get_at((0, 0))
        image.set_colorkey(color, RLEACCEL)
    return image

class mover_screen(pygame.surface.Surface):
    def __init__(self, img):
        self.img = img
        self.surf = pygame.Surface((800,600))
        self.rect = img.get_rect()
        self.rect.move_ip(0, 0)
        self.MaX = self.rect[2]
        self.MaY = self.rect[3]
        print self.MaX, self.MaY

    def mover(self):
        Mx, My = pygame.mouse.get_pos()
        x = 0
        y = 0
        if Mx < 100:
            x += 8
        if Mx > 540:
            x -= 8
        if My < 100:
            y += 8
        if My > 380:
            y -= 8
        self.rect.move_ip(Mx, My)

    def update(self,screen):
        if self.rect.x > 0:
            self.rect.x = 0
        if self.rect.x < - self.MaX:
            self.rect.x = -self.MaX
        if self.rect.y > 0:
            self.rect.y = 0
        if self.rect.y < - self.MaY:
            self.rect.y = -self.MaY
        #self.surf.blit(self.img, (self.rect.x, self.rect.y))
        screen.blit(self.img, (self.rect.x,self.rect.y))

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    img = load_image("images/Imagen_01.jpg")
    imagen = mover_screen(img)
    print img.get_rect()
    pygame.display.set_caption("Pruebas Pygame")
    while True:
        for eventos in pygame.event.get():
            if eventos.type == QUIT:
                sys.exit(0)
        imagen.mover()
        imagen.update(screen)
        pygame.display.flip()


if __name__ == '__main__':
    pygame.init()
    main()