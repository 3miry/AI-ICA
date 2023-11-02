import gymnasium as gym
import pygame
from gym import spaces
import numpy as np

class Pharmacy:
    def __init__(self, position, initial_stock, decay_rate):
        self.position = position
        self.stock = initial_stock
        self.decay_rate = decay_rate

    def decay_stock(self):
        self.stock = max(0, self.stock - self.decay_rate)  # Ensures stock doesn't go below 0

    def receive_medication(self, amount):
        self.stock += amount
        print(f"Received {amount} delivery")


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self.grid_size = 5
        # Assuming one drone for simplicity, positioned at the center
        self.drone_position = [2,2] # Initialize drone positions
        # For simplicity assuming there are 3 pharmacies at fixed positions
        self.pharmacies = [
            Pharmacy([0, 1], 50, 0.1),
            Pharmacy([0, 4], 75, 0.2),
            Pharmacy([4, 4], 100, 0.3)
        ]
        self.dc_position = [0,0]  # Assuming DC is at the top left corner
        self.past_positions = []
        self.cargo = 0  # Initialize with zero cargo
        self.max_cargo = 100  # Maximum cargo capacity
        # Define action and observation space
        self.total_reward = 0
        self.episode_count = 0
        self.action_count = 0
        self.action_space = spaces.Discrete(16)  # 4 directions + pickup + 10 dropoff actions
        self.observation_space = spaces.Box(low=0, high=self.max_cargo, shape=(9,), dtype=int)
        pygame.init()
        self.cell_size = 100
        self.text_pane_width = 200
        self.grid_pane_width = self.grid_size * self.cell_size
        self.window_width = self.text_pane_width + self.grid_pane_width
        self.window = pygame.display.set_mode((self.window_width, self.grid_pane_width))
        pygame.display.set_caption('Drone Delivery Simulation')
        self.path_surface = pygame.Surface((self.window_width, self.grid_pane_width), pygame.SRCALPHA)
        self.info_text = []


    # Update drone positions, handle pickup/dropoff, etc.
    def step(self, action):
       # print(f"Step: Action = {action}, Drone Position = {self.drone_position}, Cargo = {self.cargo}")
        reward = 0
        done = False
        info = {}

        # Get a list of pharmacy positions
        pharmacy_positions = [pharmacy.position for pharmacy in self.pharmacies]

        # Assume actions 0: up, 1: right, 2: down, 3: left, 4: pickup, 5: dropoff
        if action == 0 and self.drone_position[1] < self.grid_size - 1:
            self.drone_position[1] += 1  # Move up
        elif action == 1 and self.drone_position[0] < self.grid_size - 1:
            self.drone_position[0] += 1  # Move right
        elif action == 2 and self.drone_position[1] > 0:
            self.drone_position[1] -= 1  # Move down
        elif action == 3 and self.drone_position[0] > 0:
            self.drone_position[0] -= 1  # Move left
        elif action == 4:
            # Pickup action
            if self.drone_position == self.dc_position and self.cargo < self.max_cargo:
                self.cargo = self.max_cargo  # Pickup fully restocks drone
                reward += 20  # Reward for successful pickup, adjust as needed
        elif action == 5:
            # Dropoff action
            if self.drone_position in pharmacy_positions and self.cargo > 0:
                # Find the pharmacy object at the drone's position
                pharmacy = next(pharmacy for pharmacy in self.pharmacies if pharmacy.position == self.drone_position)
                # Update the pharmacy's stock and the drone's cargo
                pharmacy.receive_medication(1)  # Assume 1 unit of medication is dropped off
                self.cargo -= 1
                reward += 20  # Reward for successful dropoff, adjust as needed
        elif 6 <= action <= 14:
            # Dropoff multi action
            dropoff_amount = action - 5  # Actions 6-14 correspond to dropoff amounts 1-9
            if self.drone_position in pharmacy_positions and self.cargo >= dropoff_amount:
                # Find the pharmacy object at the drone's position
                pharmacy = next(pharmacy for pharmacy in self.pharmacies if pharmacy.position == self.drone_position)
                # Update the pharmacy's stock and the drone's cargo
                pharmacy.receive_medication(dropoff_amount)
                self.cargo -= dropoff_amount
                reward += 20 * dropoff_amount  # Reward for successful dropoff, adjust as needed
        elif action == 15:
            # Dropoff all action
            if self.drone_position in pharmacy_positions and self.cargo > 0:
                # Find the pharmacy object at the drone's position
                pharmacy = next(pharmacy for pharmacy in self.pharmacies if pharmacy.position == self.drone_position)
                # Update the pharmacy's stock and the drone's cargo
                pharmacy.receive_medication(self.cargo)
                reward += 30 * self.cargo  # Reward for successful dropoff, adjust as needed
                self.cargo = 0  # All cargo has been dropped off

        # Decay pharmacy stock
        for pharmacy in self.pharmacies:
            pharmacy.decay_stock()
            if pharmacy.stock == 0:
                reward -= 3  # Penalty for letting a pharmacy run out of stock

        # Assume each action has a cost of -1
        reward -= 1

        # If drone delivers medication, the episode ends
      #  if action == 5 and self.cargo > 0:
       #   done = True
        #  info['episode'] = {'success': True}

        self.current_action = action
        self.current_reward = reward
        self.total_reward += reward
        self.action_count += 1

        if self.action_count >= 1000:
            done = True
            info['episode'] = {'success': True}

        observation = self.get_observation()
       # print(f"End of Step: Reward = {reward}, Done = {done}")
        return observation, reward, done, info

    def reset(self):
        print("Resetting environment...")
        # Reset drone positions, pharmacy inventory, etc.
        self.drone_position = [2, 2]
        self.action_count = 0
        self.episode_count += 1
        # Get the current observation after resetting
        observation = self.get_observation()
        return observation

    def render(self):
        print("Rendering...")
        self.window.fill((255, 255, 255))  # White background

        # Adjusting window size and grey panel position
        text_pane_width = 200  
        grid_pane_width = self.grid_size * self.cell_size
        window_width = text_pane_width + grid_pane_width
        self.window = pygame.display.set_mode((window_width, grid_pane_width))
        pygame.draw.rect(self.window, (200, 200, 200), (0, 0, text_pane_width, grid_pane_width))  # Adjust the position and dimensions of the grey rectangle

        # Create a font object
        font = pygame.font.Font(None, 24)
                
        pygame.draw.rect(self.window, (255, 255, 255), (text_pane_width, 0, grid_pane_width, grid_pane_width))  # White background for the grid area

        # Prepare text surfaces for rendering
        self.info_text = [
            f'Total Reward: {self.total_reward}',
            f'Episode: {self.episode_count}',
            f'Actions: {self.action_count}',
            f'Action: {self.current_action}',
            f'Current Reward: {self.current_reward}'
        ]
    

        # Set a common y-coordinate offset for all text elements
        y_offset = 10  # Start rendering 10 pixels from the top of the window
        line_height = 40  # Set line height to 40 pixels, including space between lines

        for idx, line in enumerate(self.info_text):
            text_surface = font.render(line, True, (0, 0, 0))
            self.window.blit(text_surface, (10, 10 + idx * 40))  # Adjust coordinates as needed


        # Draw grid
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                rect = pygame.Rect(text_pane_width + col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.window, (0, 0, 0), rect, 1)  # Black grid lines

        # Drone rectangle
        drone_rect = pygame.Rect(
            text_pane_width + self.drone_position[0] * self.cell_size,
            self.drone_position[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.window, (0, 0, 255), drone_rect)  # Blue drone

        # Distribution Center rectangle
        dc_rect = pygame.Rect(
            text_pane_width + self.dc_position[0] * self.cell_size,
            self.dc_position[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.window, (0, 255, 0), dc_rect)  # Green distribution center

        # Pharmacies rectangles
        for pharmacy in self.pharmacies:
            pharmacy_position = pharmacy.position
            # Determine color based on stock level (for example, using a simple linear interpolation)
            red_value = max(0, min(255, 255 - int(pharmacy.stock * 2.55)))  # Clamp the red value
            green_value = max(0, min(255, int(pharmacy.stock * 2.55)))  # Clamp the green value
            color = (red_value, green_value, 0)  # Update color calculation
            pharmacy_rect = pygame.Rect(
                text_pane_width + pharmacy_position[0] * self.cell_size,
                pharmacy_position[1] * self.cell_size,
                self.cell_size, self.cell_size
            )
            pygame.draw.rect(self.window, color, pharmacy_rect)  # Color pharmacies based on stock level

            # Draw stock level text
            stock_text = font.render(f'{int(pharmacy.stock)}', True, (0, 0, 0))
            stock_text_rect = stock_text.get_rect(center=(pharmacy_rect.centerx, pharmacy_rect.centery))
            self.window.blit(stock_text, stock_text_rect.topleft)

        # Drone cargo level text
        cargo_text = font.render(f'Cargo: {self.cargo}', True, (0, 0, 0))
        self.window.blit(cargo_text, (text_pane_width + 10, 10))  # Adjust position as needed, though this might overlap with other text

        # Update past_positions list
        self.past_positions.append(self.drone_position)

##        # Draw new path segment
##        if len(self.past_positions) > 1:
##            pygame.draw.lines(
##                self.path_surface, (0, 0, 0, 51),  # 51 / 255 ≈ 20% opacity
##                False,
##                [(text_pane_width + pos[0] * self.cell_size + self.cell_size // 2, pos[1] * self.cell_size + self.cell_size // 2) for pos in self.past_positions[-2:]],  # Only draw the last segment
##                2
##            )
##
##        # Blit the path surface onto the window
##        self.window.blit(self.path_surface, (0, 0))

        pygame.display.flip()
        pygame.time.Clock().tick(60)  # This will cap the frame rate at 60 FPS, which might help reduce flickering



        pygame.display.flip()
        pygame.time.Clock().tick(60)
        print("Render complete.")

    def get_observation(self):
        # Extracting positions from the pharmacy objects
        pharmacy_positions = [pharmacy.position for pharmacy in self.pharmacies]
        # Flattening the list of pharmacy positions
        flattened_positions = [item for sublist in pharmacy_positions for item in sublist]
        # Combining drone position, cargo, and pharmacy positions into one observation
        observation = self.drone_position + [self.cargo] + flattened_positions
        return np.array(observation)




     

