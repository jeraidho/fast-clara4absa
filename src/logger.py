import csv
import os
import time

class LocalLogger:
    def __init__(self, log_dir="logs", run_name="clara_experiment"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, f"{run_name}_{int(time.time())}.csv")
        self.history = []
        self.header_written = False

    def log(self, metrics, step):
        metrics['step'] = step
        self.history.append(metrics)
        
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(metrics)

    def save_final(self):
        print(f"Logs are saved locally in: {self.filepath}")