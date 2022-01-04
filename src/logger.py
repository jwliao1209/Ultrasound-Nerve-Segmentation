import csv


class Logger():
    def __init__(self, print_=True):
        self.records = []
        self.print_ = print_

    def add(self, **inputs):
        self.records.append(inputs)
        if self.print_:
            self.print(**inputs)

        return None

    def print(self, **inputs):
        print(', '.join(f"{k}: {v}"
                        for k, v in zip(inputs.keys(), inputs.values())))

        return None

    def save(self, path):
        fieldnames = self.records[0].keys()
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        return None
