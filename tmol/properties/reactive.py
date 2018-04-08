"""
A prototype implementation of "reactive" properties in python.

A containing class stores state as properties divided into "standard" props, which represent source data, and "reactive" props, which are derived from other props.

Reactive props are calculated dynamically via a "pull" model, accessing a property invokes its definition function to calculate the property value. The value is cached within the containing object and is returned, if available, on the next property access. The definition function may access other properties within the object, which *may* be themselves reactive, resulting in a backward traversal through reactive properties to calculate the required value. DAG-ness is not mandated, but infinite recursion will occur in the case of mutually interdependent property values. Caveat usor.

Reative props are invalidated via a "push" model. Deleting or changing a property value will invalidate the store value for all declared dependent properties. These dependents may themselves have dependents, resulting in a forward traversal through reactive properties to invalidate all derived property values. This allows recalcuation when the property is next requested.

Inter-property dependencies must be *explictly* declared to support forward invalidation. Explict is better than implict, eh?
"""

import six
import properties
from toolz import get_in
from ._util import rename_code_object

class CachedProperty(properties.basic.DynamicProperty):
    def get_property(self):
        scope = self

        def fget(self):
            if not scope.name in self._backend:
                value = scope.func(self)
                if value is properties.undefined:
                    return None
                value = scope.validate(self, value)
                
                self._set(scope.name, value)
                
            return self._get(scope.name) 

        def fset(self, value):
            raise AttributeError("cannot set attribute")

        def fdel(self):
            self._set(scope.name, properties.undefined)

        return property(fget=fget, fset=fset, fdel=fdel, doc=scope.sphinx())

def cached(prop):
    def _wrap(f):
        return CachedProperty(doc=prop.doc, prop=prop, func=f)
    
    return _wrap

class DerivedProperty(properties.basic.DynamicProperty):
    @property
    def dependencies(self):
        return getattr(self, "_dependencies")
    
    @dependencies.setter
    def dependencies(self, value):
        if not isinstance(value, (tuple, list, set)):
            value = [value]
        for val in value:
            if not isinstance(val, six.string_types):
                raise TypeError('Observed names must be strings')
        self._dependencies = value
    
    def __init__(self, dependencies, doc, func, prop, **kwargs):
        self.dependencies = dependencies
        self.obs = properties.handlers.Observer(self.dependencies, "observe_set")(self.clear)
        
        properties.basic.DynamicProperty.__init__(self, doc, func, prop, **kwargs)
        
    def clear(self, instance, _):
        instance.__delattr__(self.name)
        
    def get_property(self):
        scope = self
            
        def fget(self):
            if not scope.name in self._backend:
                value = scope.func(self)
                if value is properties.undefined:
                    return None
                value = scope.validate(self, value)
                
                self._set(scope.name, value)
                
                has_observers = all(
                    any(
                        scope.obs is o 
                        for o in get_in([name, scope.obs.mode], self._listeners, ())
                    )
                    for name in scope.obs.names
                )
                
                if not has_observers:
                    properties.handlers._set_listener(self, scope.obs)
                
            return self._get(scope.name) 

        def fset(self, value):
            raise AttributeError("cannot set attribute")

        def fdel(self):
            self._set(scope.name, properties.undefined)

        suffix = "_" + scope.name

        fget = rename_code_object(fget, "fget_" + scope.name)
        fset = rename_code_object(fset, "fset_" + scope.name)
        fdel = rename_code_object(fdel, "fdel_" + scope.name)

        return property(fget=fget, fset=fset, fdel=fdel, doc=scope.sphinx())
    
def derived_from(dependencies, prop):
    def _wrap(f):
        return DerivedProperty(dependencies, doc=prop.doc, prop=prop, func=f)
    
    return _wrap
