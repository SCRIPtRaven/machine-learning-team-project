class PIDController:
    def __init__(self, kp, ki, kd, setpoint1, setpoint2, weight_v1, weight_v2, feedforward_gain=1.0, min_output=0, max_output=1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint1 = setpoint1
        self.setpoint2 = setpoint2
        self.weight_v1 = weight_v1
        self.weight_v2 = weight_v2
        self.feedforward_gain = feedforward_gain
        self.min_output = min_output
        self.max_output = max_output
        self.prev_error = 0
        self.integral = 0

    def compute(self, current_value_v1, current_value_v2, feedforward_input):
        error1 = self.setpoint1 - current_value_v1
        error2 = self.setpoint2 - current_value_v2
        combined_error = (self.weight_v1 * error1 + self.weight_v2 * error2) / (self.weight_v1 + self.weight_v2)

        self.integral += combined_error
        derivative = combined_error - self.prev_error

        # Anti-windup: Clamp integral to prevent excessive accumulation
        self.integral = max(-self.max_output / self.ki, min(self.max_output / self.ki, self.integral))

        # Feedforward control
        feedforward = self.feedforward_gain * feedforward_input

        # PID formula with feedforward
        output = self.kp * combined_error + self.ki * self.integral + self.kd * derivative + feedforward
        output = max(self.min_output, min(self.max_output, output))  # Clamp output to [min_output, max_output]

        self.prev_error = combined_error

        return output

    def adapt_parameters(self, error):
        if abs(error) > 0.1:
            self.kp += 0.01 * error
            self.ki += 0.005 * error
            self.kd += 0.005 * error
        else:
            self.kp -= 0.01 * error
            self.ki -= 0.005 * error
            self.kd -= 0.005 * error

        # Clamp PID parameters to reasonable values
        self.kp = max(0.1, min(2.0, self.kp))
        self.ki = max(0.01, min(0.5, self.ki))
        self.kd = max(0.01, min(0.5, self.kd))
