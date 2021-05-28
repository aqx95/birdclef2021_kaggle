class MetricMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {"loss": 0, "lrap":0, "precision":0, "recall":0, "f1":0}
        self.avg_metric = {"loss": 0, "lrap":0, "precision":0, "recall":0, "f1":0}
        self.sum = 0

    def update(self, metric, batch_size=1):
        for key, values in metric.items():
            self.metrics[key] += values*batch_size
        self.sum += batch_size

    @property
    def avg(self):
        for key, values in self.metrics.items():
            self.avg_metric[key] = self.metrics[key] / self.sum
        return self.avg_metric
