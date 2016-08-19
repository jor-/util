import util.logging
logger = util.logging.logger


class Observable:
    
    def __init__(self, observers=None):
        self.observers = []
        if observers is None:
            observers = []
        for observer in observers:
            self.add_observer(observer)
    
    def add_observer(self, observer):
        logger.debug('{}: observer {} added.'.format(self.__class__.__name__, observer))
        self.observers.append(observer)
    
    def remove_observer(self, observer):
        logger.debug('{}: observer {} removed.'.format(self.__class__.__name__, observer))
        self.observers.remove(observer)
    
    
    def _notify_observers(self, notify_method):
        for observer in self.observers:
            notify_method(observer)
    
    def notify_changed(self, key, new_value, old_value):
        logger.debug('{}: notify changed: key {}, value {}, old value {}.'.format(self.__class__.__name__, key, value, old_value))
        self._notify_observers(lambda observer: observer.notify_changed(key, new_value, old_value))
    
    def notify_added(self, key, new_value):
        logger.debug('{}: notify add: key {}, value {}.'.format(self.__class__.__name__, key, value))
        self._notify_observers(lambda observer: observer.notify_added(key, new_value))
    
    def notify_removed(self, key, old_value):
        logger.debug('{}: notify del: key {}, old value {}.'.format(self.__class__.__name__, key, old_value))
        self._notify_observers(lambda observer: observer.notify_removed(key, old_value))