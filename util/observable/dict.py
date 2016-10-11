import util.observable.base

import util.logging
logger = util.logging.logger



class ObservableDict(util.observable.base.Observable, dict):
    
    def __init__(self, observers=None):
        dict.__init__(self)
        util.observable.base.Observable.__init__(self, observers=observers)
    
    
    def __setitem__(self, key, value):
        try:
            old_value = self.__getitem__(key)
        except KeyError:
            super().__setitem__(key, value)
            self.notify_added(key, value)
        else:
            if value != old_value:
                super().__setitem__(key, value)
                self.notify_changed(key, value, old_value)
    
    
    def __delitem__(self, key):
        value = self.__getitem__(key)
        super().__delitem__(key)
        self.notify_removed(key, value)
    
    
    def clear(self):
        old_dict = self.copy()
        super().clear()
        for key, value in old_dict.items():
            self.notify_removed(key, value)
    
    
    def update(self, update_dict):
        changed_keys_values = []
        new_keys_values = []
        for key, value in update_dict.items():
            try:
                old_value = self[key]
            except KeyError:
                new_keys_values.append((key, value))
            else:
                if value != old_value:
                    changed_keys_values.append((key, value, old_value))
        
        super().update(update_dict)
        
        for (key, value, old_value) in changed_keys_values:
            self.notify_changed(key, value, old_value)        
        for (key, value) in new_keys_values:
            self.notify_added(key, value)
    
    
    def setdefault(self, key, value=None):
        value_added = key not in self
        value = super().setdefault(key, value)
        if value_added:
            self.notify_added(key, value)
        return value

    
    def pop(self, k, value=None):
        value_removed = key in self
        value = super().pop(self, k, value)
        if value_removed:
            self.notify_removed(key, value)
        return value

    
    def popitem(self):
        key, value = dict.popitem(self)
        self.notify_removed(key, value)
        return key, value 
    


class DependencyObserver:
    
    def __init__(self, independent_observable_dict, dependent_observable_dict, independent_key=None, dependent_key=None):
        self.independent_observable_dict = independent_observable_dict
        self.dependent_observable_dict = dependent_observable_dict
        self.independent_key = independent_key
        self.dependent_key = dependent_key
        
        independent_observable_dict.add_observer(self)
    
    
    def notify_dependency(self, key):
        if self.independent_key is None or key == self.independent_key:
            logger.debug('{}: dependency activated: independent dict {}, independent key {}, dependent dict {}, dependent key {}.'.format(self.__class__.__name__, self.independent_observable_dict, self.independent_key, self.dependent_observable_dict, self.dependent_key))
            if dependent_key is not None:
                del self.dependent_observable_dict[self.dependent_key]
            else:
                self.dependent_observable_dict.clear()
    
    
    def notify_changed(self, key, new_value, old_value):
        self.notify_dependency(key)
    
    def notify_added(self, key, new_value):
        self.notify_dependency(key)
    
    def notify_removed(self, key, old_value):
        self.notify_dependency(key)


