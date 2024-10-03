class ActionSet():
    def __init__(self, actions = None):
        if not actions:
            self.actions = []
        elif not isinstance(actions, list):
            self.actions = [actions]

    def add_action(self, action):
        self.actions.append(action)

class MergeAction():
    def __init__(self, keep_object, discard_object):
        self.keep_object = keep_object
        self.discard_object = discard_object

class AddSliceAction():
    def __init__(self, object, slice):
        self.object = object
        self.slice = slice

class ProjectionAction():   
    def __init__(self, xy_list, xz_list, xy_slice, xz_slice, outcome):
        self.xy_slice = xy_slice
        self.xz_slice = xz_slice
        self.xy_list = xy_list
        self.xz_list = xz_list
        self.outcome = outcome
