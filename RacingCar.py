import math
import time
import pygame
import neat
import time
import sys

WIN_WIDTH = 900
WIN_HEIGHT = 750

Red_Car = pygame.transform.scale_by(pygame.image.load(r'E:\Python\ML\Game\RacingCar\RedCar.png'), 1.0)
White_Car = pygame.transform.scale_by(pygame.image.load(r'E:\Python\ML\Game\RacingCar\WhiteCar.png'), 1.1)

Track_Border = pygame.transform.scale_by(pygame.image.load(r'E:\Python\ML\Game\RacingCar\imgs\track-border.png'), 0.85)
Track_Borderorder_mask =  pygame.mask.from_surface(Track_Border)
Track = pygame.transform.scale_by(pygame.image.load(r'E:\Python\ML\Game\RacingCar\imgs\track.png'), 0.85)

Finish_Line = pygame.image.load(r'E:\Python\ML\Game\RacingCar\imgs\finish.png')
Finish_Line_mask =  pygame.mask.from_surface(Finish_Line)
Grass = pygame.transform.scale(pygame.image.load(r'E:\Python\ML\Game\RacingCar\imgs\grass.jpg'), (WIN_WIDTH, WIN_HEIGHT))

pygame.display.set_caption('Racing Game by AI!')

def blit_rotate_center(win, image, top_left, angle):
     '''
     Rotate the car around its centre without changing the x's and y's position
     '''
     rotated_image = pygame.transform.rotate(image, angle)
     new_rect = rotated_image.get_rect(center= image.get_rect(topleft = top_left).center)
     win.blit(rotated_image, new_rect.topleft)
     
class Properties:
     def __init__(self, max_vel, rotation_vel):
          self.image = self.IMG 
          self.max_vel = max_vel
          self.vel = 0
          self.rotation_vel = rotation_vel
          self.angle = 0
          self.x, self.y = self.START_POS
          self.acceleration = 0.2
          
     def rotate(self, Left=False, Right=False):
          if Left:
               self.angle += self.rotation_vel
               
          elif Right: 
               self.angle -= self.rotation_vel
     
     def move_forward(self):
          self.vel = min(self.vel + self.acceleration, self.max_vel)
          self.move()  
            
     def move(self):
          radians = math.radians(self.angle)
          horizontal = math.cos(radians) * self.vel
          vertical = math.sin(radians) * self.vel
          
          self.x -= horizontal
          self.y -= vertical
     
     def slow_down(self):
          self.vel = max(self.vel - self.acceleration / 2 , 0)
          self.move()
          
     def draw(self, win):
          blit_rotate_center(win, self.image, (self.x, self.y), self.angle)
          
     def collision(self, mask, x=0, y=0):
          
          car_mask = pygame.mask.from_surface(self.image)
          
          offset = (int(self.x - x), int(self.y - y))
          
          collide_point = mask.overlap(car_mask, offset)
          
          if collide_point: 
               return True
          
          return False
         
          
class Car(Properties):
     IMG = White_Car
     START_POS = (100, 170)
     
          
def draw_window(win, cars):
     win.blit(Grass, (0, 0))
     win.blit(Track, (70, 0))
     win.blit(Finish_Line, (80, 270))
     win.blit(Track_Border, (70, 0))
     
     for car in cars:
          car.draw(win)
     
     pygame.display.update()
     
def main(genomes, config):
     nets = []
     ge = []
     cars = []
     
     for _, genome in genomes: 
          net = neat.nn.FeedForwardNetwork.create(genome, config)
          nets.append(net)
          cars.append(Car(4,4))
          genome.fitness = 0
          ge.append(genome)
     
     win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
     car = Car(4,4)
     
     run = True
     clock = pygame.time.Clock()
     
     FPS = 60
     while run:
          clock.tick(FPS) 
          for event in pygame.event.get():
               if event.type == pygame.QUIT:
                    run = False
                    pygame.quit() 
                    quit()
          
          for x, car in enumerate(cars):
               ge[x].fitness += 0.2
               car.move()
               
               left = 0
               right = 0
               in_front = 0 
               left_45 = 0
               right_45 = 0
               
               
               output = nets[cars.index(car)].activate((left, right, in_front, left_45, right_45))
               
               if output[0] > 0.5:
                    pass 
                    
               if output[1] > 0.5:
                    pass
               
               if output[2] > 0.5:
                    pass
               
               if output[3] > 0.5:
                    pass              
                    
          ## Check for collision
          '''
          for x, car in enumerate(cars):
          
          if car.collision(Track_Borderorder_mask):
               ge[x].fitness -= 2
               cars.pop(x)
               nets.pop(x)
               ge.pop(x)
               
          elif car.collision(Finish_Line_mask):
               for g in ge: 
                    g.fitness += 10
          '''                  
          draw_window(win, car)
          
    
main()