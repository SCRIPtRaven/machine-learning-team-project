class PIDController:
    def __init__(self, kp, ki, kd, setpoint1, setpoint2, weight_v1, weight_v2, min_output=0, max_output=1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint1 = setpoint1
        self.setpoint2 = setpoint2
        self.weight_v1 = weight_v1
        self.weight_v2 = weight_v2
        self.min_output = min_output
        self.max_output = max_output
        self.prev_error = 0
        self.integral = 0

    def compute(self, current_value_v1, current_value_v2):
        # Combined weighted error
        error = (self.weight_v1 * (self.setpoint1 - current_value_v1) + self.weight_v2 * (self.setpoint2 - current_value_v2)) / (self.weight_v1 + self.weight_v2)
        self.integral += error
        derivative = error - self.prev_error

        # PID formula
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(self.min_output, min(self.max_output, output))  # Clamp output to [min_output, max_output]

        self.prev_error = error

        return output
