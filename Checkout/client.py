import requests
import time
from detector import *
from generator import *
import typer

class WeightClient:
    def __init__(self, detector, server_url):
        self.detector = detector
        self.server_url = server_url

    def send_stable_values(self, sensor_values):
        '''
        Method indicating whether the sensor data was successfully transmitted or not.
        '''
        
        response = requests.post(self.server_url, json={'sensor_values': sensor_values})
        if response.status_code == 200:
            print(f"Stable values sent to server: {sensor_values}")
        else:
            print(f"Failed to send stable values. Status code: {response.status_code}")

    def monitor_weight_changes(self, generator, detector, min_steps_stab=3, initial_weight = 0):
        '''
        A method that generates sensor weight and verifies for any changes. If a change is detected, it checks for stability, and if confirmed, the values are then transmitted to the server.
        '''
        current_weight = [initial_weight for _ in range(generator.number_sensors)]
        while True:
            try:
                generator.generate()
            except AssertionError as e:
                print(e)
                exit()
            new_weights = generator.current_weights

            check_change, _ = detector.check_weight_change(current_weight, new_weights)
            if check_change:
                for _ in range(min_steps_stab):
                    try:
                        generator.generate()
                    except AssertionError as e:
                        print(e)
                        exit()
                    total_sum = sum(generator.current_weights)
                    if not detector.check_stability(total_sum):
                        break

                if detector.is_stable(min_steps_stab):
                    self.send_stable_values(new_weights.tolist())
                    detector.reset()

            time.sleep(1)

            current_weight = new_weights

            


def run_code(
    weightSensorPath: str = typer.Option(
        '/home/anca/Desktop/Checkout/data/weights.txt',
        "--weight-sensor-path",
        help="Path to the file that contains the weight of the sensors",
    ),
    stabIntervalInputPath: str = typer.Option(
        '/home/anca/Desktop/Checkout/data/stab_interval.txt',
        "--stable-intervals-path",
        help="Path to the file that contains the values of the stable intervals.",
    ),
    serverUrl: str = typer.Option(
        'http://192.168.1.233:5000/determine_change/',
        "--server-url",
        help="The server url the client sends a post request to.",
    ),
    weightThrs: int = typer.Option(
        10,
        "--weight-treshold",
        help="The value of the weight threshold for checking for any change.",
    )):

    generator = WeightSensors(weightSensorPath)
    detector = WeightDetector(weightThrs, stabIntervalInputPath)

    client = WeightClient(detector, serverUrl)
    client.monitor_weight_changes(generator, detector)


if __name__ == "__main__":
    typer.run(run_code)
