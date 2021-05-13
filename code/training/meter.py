class MetricMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}
        self.sum = 0

    def update(self, metric, batch_size=1):
        for key, values in metric.items():
            self.metrics[key] += values*batch_size
        self.sum += batch_size

    @property
    def avg(self):
        for key, values in self.metrics.item():
            self.metrics[key] /= self.sum
        return self.metrics
