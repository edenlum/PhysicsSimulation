# Theory:
# We are making a physics simulation of a fluid. The fluid is represented by a 2D grid of cells without particles.
# Each cell is a square with side d, has density rho. We calculate the mass of the fluid in each cell as rho*d^2.
# The pressure in each cell can be calculated by rho*temperature. 
# We calculate the velocity of the fluid on the faces of each cell and not in the center. 
# 
# _________Vv_________________Vv_________________Vv_______
# |                  |                  |                 |
# |                  |                  |                 |
# |                  |                  |                 |
# >       rho        >       rho        >       rho       >
# v                  v                  v                 v
# |                  |                  |                 |
# |                  |                  |                 |
# |________Vv________|________Vv________|________Vv_______|
#
# The acceleration of the fluid on the faces of each cell is calculated by the force divided by the mass.
# a = F/m = delta P / (rho*d) = (P1 - P2) / (rho*d)
# v_t = v_t-1 + a*dt
# Because the particles move, the velocity field changes in the direction of the velocity.
# To advance the velocity in the x direction we calculate the velocity in the y direction at the same position, 
# then we calculate the previous position by subtracting the velocity*dt
# and then we interpolate the velocity in the x direction at the previous position and assign it to the current position.
# 
# In conclusion, we save a density field, and velocity field for the faces. 
# On the boudaries we have a 0 velocity field.
# The density field is updated by the velocity field: 
# density[:, :] += (velx[:-1, :] - velx[1:, :]) * dt
# density[:, :] += (vely[:, :-1] - vely[:, 1:]) * dt

import pygame
import numpy as np
import sys
import pygame_gui
import scipy

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1500, 900  # Window dimensions
d = 10  # Size of each grid cell
ROWS, COLS = HEIGHT // d, WIDTH // d
dt = 0.1  # time step
epsilon = 0.001
gamma = 0.999
wall_or_wind = 'wall'
GRAVITY = 0.0
pause = False
compressible = True
wind_tunnle = False
n_iters = 10

# Initialize the GUI manager
gui_manager = pygame_gui.UIManager((WIDTH, HEIGHT))

# Create the UI elements
add_wall_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 10), (100, 50)), text='Add Wall', manager=gui_manager)
add_wind_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((120, 10), (100, 50)), text='Add Wind', manager=gui_manager)
reset_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((230, 10), (100, 50)), text='Reset', manager=gui_manager)
incompressible_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((340, 10), (100, 50)), text='Incompressible', manager=gui_manager)
push_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((450, 10), (100, 50)), text='Push', manager=gui_manager)
smoke_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((560, 10), (100, 50)), text='Smoke', manager=gui_manager)
wind_tunnel_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((670, 10), (100, 50)), text='Wind Tunnel', manager=gui_manager)
gravity_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((10, 70), (200, 20)), start_value=GRAVITY, value_range=(0, 0.05), manager=gui_manager)

# Create window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Fluid Simulation')

# Create a 2D NumPy array with random values between 0 and 255
def reset():
    global density, smoke_density, velx, vely, walls, wind, zero_velx, zero_vely
    density = np.random.rand(ROWS, COLS)/10 + 0.5
    smoke_density = np.zeros((ROWS, COLS))
    velx = np.zeros((ROWS, COLS+1))/5
    vely = np.zeros((ROWS+1, COLS))
    walls = np.zeros((ROWS, COLS), dtype=bool)
    zero_velx = np.zeros((ROWS, COLS+1), dtype=bool)
    zero_vely = np.zeros((ROWS+1, COLS), dtype=bool)
    wind = np.zeros((ROWS, COLS), dtype=bool)
    
    zero_velx[:, 0] = True
    zero_velx[:, -1] = True
    zero_vely[0, :] = True
    zero_velx[-1, :] = True

# Map 2D data to 3D color (R, G, B)
color_data = np.zeros((ROWS, COLS, 3), dtype=np.uint8)

def zero_walls(velx, vely):
    global zero_velx, zero_vely
    velx[zero_velx] = 0
    vely[zero_vely] = 0
    if wind_tunnle:
        velx[35:55, 0] = 2
        velx[:, -1] = 2

def add_wind(force_x, force_y):
    force_x[wind[:, :-1]] += 0.5

def calc_pressure(density):
    return density

def calc_acceleration(density, velx, vely, gravity=GRAVITY):
    pressure = calc_pressure(density)
    force_x = (pressure[:, :-1] - pressure[:, 1:])
    force_y = (pressure[:-1, :] - pressure[1:, :] + gravity)
    add_wind(force_x, force_y)

    velx[:, 1:-1] +=  force_x/ (d * (density[:, :-1] + density[:, 1:] + epsilon)/2) * dt
    vely[1:-1, :] +=  force_y/ (d * (density[:-1, :] + density[1:, :] + epsilon)/2) * dt
    
    zero_walls(velx, vely)
    if not compressible:
        for n in range(n_iters):
            divergence = (velx[:, 1:] - velx[:, :-1] + vely[1:, :] - vely[:-1, :]) 
            num_of_non_wall_faces = 4 - zero_velx[:, 1:] - zero_velx[:, :-1] - zero_vely[1:, :] - zero_vely[:-1, :]
            
            divergence *= 1
            divergence /= np.maximum(1, num_of_non_wall_faces)

            velx[:, 1:-1] -= divergence[:, :-1] 
            velx[:, 1:-1] += divergence[:, 1:]
            vely[1:-1, :] -= divergence[:-1, :]
            vely[1:-1, :] += divergence[1:, :]

            zero_walls(velx, vely)

    zero_walls(velx, vely)
    # regularize velocity
    # velx[:, 1:-1] = np.clip(velx[:, 1:-1], -1+epsilon, 1-epsilon)
    # vely[1:-1, :] = np.clip(vely[1:-1, :], -1+epsilon, 1-epsilon)

def calc_density(density, velx, vely):
    global smoke_density
    if not wind_tunnle:
        padded_density = np.pad(density, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        padded_smoke_density = np.pad(smoke_density, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    else:
        padded_density = np.pad(density, ((1, 1), (1, 1)), mode='constant', constant_values=((0, 0), (1, 0)))
        padded_smoke_density = np.pad(smoke_density, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    density[:, :] += (np.maximum(0, velx[:, :-1])*padded_density[1:-1, :-2] - np.maximum(0, -velx[:, :-1])*padded_density[1:-1, 1:-1] \
                     -np.maximum(0, velx[:, 1:])*padded_density[1:-1, 1:-1] + np.maximum(0, -velx[:, 1:])*padded_density[1:-1, 2:]) * dt
    density[:, :] += (np.maximum(0, vely[:-1, :])*padded_density[:-2, 1:-1] - np.maximum(0, -vely[:-1, :])*padded_density[1:-1, 1:-1] \
                     -np.maximum(0, vely[1:, :])*padded_density[1:-1, 1:-1] + np.maximum(0, -vely[1:, :])*padded_density[2:, 1:-1]) * dt
    smoke_density[:, :] += (np.maximum(0, velx[:, :-1])*padded_smoke_density[1:-1, :-2] - np.maximum(0, -velx[:, :-1])*padded_smoke_density[1:-1, 1:-1] \
                           -np.maximum(0, velx[:, 1:])*padded_smoke_density[1:-1, 1:-1] + np.maximum(0, -velx[:, 1:])*padded_smoke_density[1:-1, 2:]) * dt
    smoke_density[:, :] += (np.maximum(0, vely[:-1, :])*padded_smoke_density[:-2, 1:-1] - np.maximum(0, -vely[:-1, :])*padded_smoke_density[1:-1, 1:-1] \
                           -np.maximum(0, vely[1:, :])*padded_smoke_density[1:-1, 1:-1] + np.maximum(0, -vely[1:, :])*padded_smoke_density[2:, 1:-1]) * dt

def calc_next_velocity(density, velx, vely):
    # We use semi lagrangian advection
    # to update velx[i, j] we first interpolate vely at the same position, 
    # then we calculate the previous position by subtracting the velocity*dt
    # and then we interpolate velx at the previous position and assign it to velx[i, j]
    x = np.arange(0, COLS+1)
    y = np.arange(0, ROWS) + 0.5
    xm, ym = np.meshgrid(x, y)
    vely_at_velx = (vely[:-1, :-1] + vely[:-1, 1:] + vely[1:, :-1] + vely[1:, 1:])/4
    spline = scipy.interpolate.RectBivariateSpline(y, x, velx, kx=1, ky=1)
    velx[:, 1:-1] = spline.ev(ym[:, 1:-1] - vely_at_velx * dt, xm[:, 1:-1] - velx[:, 1:-1] * dt) * gamma

    x = np.arange(0, COLS) + 0.5
    y = np.arange(0, ROWS+1)
    xm, ym = np.meshgrid(x, y)    
    velx_at_vely = (velx[:-1, :-1] + velx[:-1, 1:] + velx[1:, :-1] + velx[1:, 1:])/4
    spline = scipy.interpolate.RectBivariateSpline(y, x, vely, kx=1, ky=1)
    vely[1:-1, :] = spline.ev(ym[1:-1, :] - vely[1:-1, :] * dt, xm[1:-1, :] - velx_at_vely * dt) * gamma

def update():
    global density, smoke_density, velx, vely

    calc_acceleration(density, velx, vely, gravity=GRAVITY)
    calc_density(density, velx, vely)
    calc_next_velocity(density, velx, vely)

    density = np.clip(density, epsilon, 1-epsilon)
    smoke_density = np.clip(smoke_density, epsilon, 1-epsilon)
    color_data[:, :, 0] = (1 - density) * 255 # Red channel
    color_data[:, :, 1] = np.clip(smoke_density * 255*2, 0, 255) # Green channel
    color_data[:, :, 2] = density * 255 # Blue channel
    color_data[walls, :] = [100, 100, 100]
    
    # Create a Pygame surface from the NumPy array
    image_surface = pygame.surfarray.make_surface(np.transpose(color_data, (1, 0, 2)))
    return image_surface

# Main loop
reset()
while True:
    if not pause:
        image_surface = update()
    screen.blit(pygame.transform.scale(image_surface, (WIDTH, HEIGHT)), (0, 0))

    for event in pygame.event.get():   
        # Process GUI events
        gui_manager.process_events(event)
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if pygame.mouse.get_pressed()[0]:
            try:
                mouse_pos = pygame.mouse.get_pos()
            except AttributeError:
                pass
            if add_wall_button.rect.collidepoint(mouse_pos):
                wall_or_wind = 'wall'
            elif add_wind_button.rect.collidepoint(mouse_pos):
                wall_or_wind = 'wind'
            elif push_button.rect.collidepoint(mouse_pos):
                wall_or_wind = 'push'
            elif smoke_button.rect.collidepoint(mouse_pos):
                wall_or_wind = 'smoke'
            elif incompressible_button.rect.collidepoint(mouse_pos):
                compressible = not compressible
            elif wind_tunnel_button.rect.collidepoint(mouse_pos):
                wind_tunnle = not wind_tunnle
            elif reset_button.rect.collidepoint(mouse_pos):
                reset()
            elif gravity_slider.rect.collidepoint(mouse_pos):
                GRAVITY = gravity_slider.get_current_value()
            else:
                i, j = np.clip(mouse_pos[1] // d, 0, ROWS-1), np.clip(mouse_pos[0] // d, 0, COLS-1)
                if wall_or_wind=='wall': 
                    walls[i, j] = True 
                    zero_velx[i, j] = True
                    zero_velx[i, j+1] = True
                    zero_vely[i, j] = True
                    zero_vely[i+1, j] = True
                elif wall_or_wind=='push':
                    velx[i, j] = 0.5
                    velx[i, j+1] = 0.5
                    vely[i, j] = 0.5
                    vely[i+1, j] = 0.5
                elif wall_or_wind=='smoke':
                    smoke_density[i, j] = 0.5
                    smoke_density[i, j+1] = 0.5
                    smoke_density[i+1, j] = 0.5
                    smoke_density[i+1, j+1] = 0.5
                    smoke_density[i-1, j] = 0.5
                    smoke_density[i-1, j+1] = 0.5
                    smoke_density[i, j-1] = 0.5
                    smoke_density[i+1, j-1] = 0.5
                    smoke_density[i-1, j-1] = 0.5

                else: 
                    wind[i, j] = True

        # pause simulatio on space
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            pause = not pause
       
    gui_manager.update(dt)
    gui_manager.draw_ui(screen)
    pygame.display.update()

