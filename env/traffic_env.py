import random
import numpy as np

class TrafficIntersectionEnv:
    def __init__(self, arrival_rate=0.3, max_queue=10, phase_duration=5):
        """
        4 roads (N,S,E,W), each with 2 lanes:
          - Incoming lane (cars arriving towards intersection)
          - Outgoing lane (cars leaving intersection, no traffic light control)
        Traffic light controls only incoming lanes:
          - NS green, EW red or vice versa
        """
        self.roads = ['N', 'S', 'E', 'W']
        self.lanes = ['incoming', 'outgoing']
        self.arrival_rate = arrival_rate
        self.max_queue = max_queue
        self.phase_duration = phase_duration

        # Queues per lane: {road: {lane: queue_length}}
        self.queues = {r: {'incoming': 0, 'outgoing': 0} for r in self.roads}

        # Traffic light state: "NS_green" or "EW_green"
        self.traffic_light = "NS_green"
        self.time_in_phase = 0

        self.total_waiting_time = 0
        self.total_cars_passed = 0
        self.step_count = 0

    def reset(self):
        self.queues = {r: {'incoming': 0, 'outgoing': 0} for r in self.roads}
        self.traffic_light = "NS_green"
        self.time_in_phase = 0
        self.total_waiting_time = 0
        self.total_cars_passed = 0
        self.step_count = 0
        return self._get_state()

    def step(self, action=None):
        """
        One timestep in the environment.
        action: Optional, override traffic light phase switch if provided ('NS_green' or 'EW_green')
        """
        self.step_count += 1

        # 1. Car arrivals on incoming lanes (random)
        for road in self.roads:
            if self.queues[road]['incoming'] < self.max_queue:
                if random.random() < self.arrival_rate:
                    self.queues[road]['incoming'] += 1

        # 2. Update traffic light phase if action given, else automatic switch after phase_duration
        if action in ['NS_green', 'EW_green']:
            self.traffic_light = action
            self.time_in_phase = 0
        else:
            self.time_in_phase += 1
            if self.time_in_phase >= self.phase_duration:
                self.traffic_light = "EW_green" if self.traffic_light == "NS_green" else "NS_green"
                self.time_in_phase = 0

        # 3. Move cars through intersection if light green
        if self.traffic_light == "NS_green":
            for road in ['N', 'S']:
                if self.queues[road]['incoming'] > 0:
                    self.queues[road]['incoming'] -= 1
                    self.queues[road]['outgoing'] = min(self.queues[road]['outgoing'] + 1, self.max_queue)
                    self.total_cars_passed += 1
        else:
            for road in ['E', 'W']:
                if self.queues[road]['incoming'] > 0:
                    self.queues[road]['incoming'] -= 1
                    self.queues[road]['outgoing'] = min(self.queues[road]['outgoing'] + 1, self.max_queue)
                    self.total_cars_passed += 1

        # 4. Outgoing lanes cars leave the system (free flow)
        for road in self.roads:
            self.queues[road]['outgoing'] = max(self.queues[road]['outgoing'] - 1, 0)

        # 5. Calculate total waiting time (sum of incoming queues)
        waiting_cars = sum(self.queues[road]['incoming'] for road in self.roads)
        self.total_waiting_time += waiting_cars

        # 6. Compose state and reward
        state = self._get_state()
        reward = -waiting_cars  # negative waiting cars as reward (minimize waiting)
        done = False  # continuous env, no terminal state

        return state, reward, done, {}

    def _get_state(self):
        # State as a vector: [incoming_N, outgoing_N, incoming_S, outgoing_S, incoming_E, outgoing_E, incoming_W, outgoing_W, traffic_light]
        state_list = []
        for road in self.roads:
            state_list.append(self.queues[road]['incoming'])
            state_list.append(self.queues[road]['outgoing'])
        # Encode traffic light as 0 for NS_green, 1 for EW_green
        light_enc = 0 if self.traffic_light == "NS_green" else 1
        state_list.append(light_enc)
        return np.array(state_list, dtype=np.float32)

    def render(self):
        print(f"Step: {self.step_count} | Traffic light: {self.traffic_light}")
        for road in self.roads:
            inc = self.queues[road]['incoming']
            out = self.queues[road]['outgoing']
            print(f"  Road {road}: Incoming={inc}, Outgoing={out}")
        print(f"Total cars passed: {self.total_cars_passed}, Total waiting time: {self.total_waiting_time}")
        print("-" * 40)
