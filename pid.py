"""
Šitą balaganą galimai reikėtų perrašyti
"""


class PIDController:
    def __init__(self, kp, ki, kd, setpoint1, setpoint2, weight_v1, weight_v2, weight_combined, min_output=0,
                 max_output=1, smoothing_factor=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint1 = setpoint1
        self.setpoint2 = setpoint2
        self.weight_v1 = weight_v1
        self.weight_v2 = weight_v2
        self.weight_combined = weight_combined
        self.min_output = min_output
        self.max_output = max_output
        self.prev_error1 = 0
        self.prev_error2 = 0
        self.integral1 = 0
        self.integral2 = 0
        self.smoothing_factor = smoothing_factor
        self.prev_output = 0

    def compute(self, current_value_v1, current_value_v2, rt, intermediate_setpoint=None):
        if intermediate_setpoint is None:
            error1 = self.setpoint1 - current_value_v1
            error2 = self.setpoint2 - current_value_v2
        else:
            error1 = intermediate_setpoint - current_value_v1
            error2 = intermediate_setpoint - current_value_v2

        combined_error = self.weight_combined * (self.weight_v1 * error1 + self.weight_v2 * error2 + rt)
        weighted_error1 = self.weight_v1 * error1
        weighted_error2 = self.weight_v2 * error2

        self.integral1 += error1
        self.integral2 += error2
        derivative1 = error1 - self.prev_error1
        derivative2 = error2 - self.prev_error2

        raw_output1 = self.kp * weighted_error1 + self.ki * self.integral1 + self.kd * derivative1
        raw_output2 = self.kp * weighted_error2 + self.ki * self.integral2 + self.kd * derivative2
        raw_output_combined = self.kp * combined_error + self.ki * (self.integral1 + self.integral2) + self.kd * (
                derivative1 + derivative2)

        raw_output = self.weight_v1 * raw_output1 + self.weight_v2 * raw_output2 + self.weight_combined * raw_output_combined
        smoothed_output = self.smoothing_factor * raw_output + (1 - self.smoothing_factor) * self.prev_output

        smoothed_output = max(self.min_output, min(self.max_output, smoothed_output))

        self.prev_output = smoothed_output
        self.prev_error1 = error1
        self.prev_error2 = error2

        return smoothed_output
