from abc import ABC, abstractmethod

class DataProcessor(ABC):
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def process_all_files(self):
        pass

    def process(self):
        data = self.load_data()
        self.process_all_files(data)
