# Safe_protect_module
# Author: Yuansj
# Time : 2022/03/08

class safe_check:
    '''
    A rule based module to ensure safety, if ego want to change lane, 
    but there have been a vehicle in the target lane. Then reject the action, choose slower to make sure safety.
    '''
    def __init__(self, road, ego) -> None:
        self.ego = ego
        self.road = road
        self.all_action = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
        self.dt = 0.1
        self.N_1 = 3
        self.N_2 = 8

    def collision_check(self, action:int) -> bool:
        choose_action = self.all_action[action]
        collision_other, collision_object, wrong_action = False, False, False
        future_state = self.ego.safe_act(choose_action)
        ego_future_positionx = future_state[(self.N_1+1)*4,0]
        ego_future_positiony = future_state[(self.N_1+1)*4+1,0]

        if self.ego.position[0] < self.road.road_ends and self.ego.position[0] > 630:
            for i in range(len(self.road.vehicles) - 1):
                current_position = self.road.vehicles[i].position
                current_speed = self.road.vehicles[i].speed

                # check collision with other vehicles
                position1 = current_position[0] + current_speed * self.dt * self.N_1
                position2 = current_position[0] + current_speed * self.dt * self.N_2
                
                if ego_future_positionx > position1 and ego_future_positionx < position2 and \
                    abs(ego_future_positiony - current_position[1]) < 2:
                    collision_other = True
                
                # collision with objects
                if ego_future_positionx > self.road.road_ends - 30 and ego_future_positiony > 4.8:
                    collision_object = True

                # wrong action
                if self.ego.position[1] < 5/2 and action == 2:
                    wrong_action = True
        
        unsafe_action = [collision_other, collision_object, wrong_action]
        return unsafe_action
    
    def correct_action(self, unsafe_action, action:int):
        correct_action = action
        # if self.ego.position[1] > 2.5 and self.ego.position[0] > 630:
        #     check_if_merge = self.collision_check(0)
        #     if check_if_merge[0] == False:
        #         correct_action = 0
        #     elif check_if_merge[0] == True:
        #         correct_action = 4
        
        # else:
        #     if unsafe_action[2] == True:
        #         correct_action = 1
        #     elif unsafe_action[0] == True:
        #         correct_action = 4

        if unsafe_action[2] == True:
            correct_action = 1
        if unsafe_action[0] == True:
            correct_action = 4
        if unsafe_action[1] == True:
            correct_action = 4

        return correct_action