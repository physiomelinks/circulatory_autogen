'''
Created on 29/10/2021

@author: Gonzalo D. Maso Talou, Finbar Argus
'''

class CVS0DModel(object):
    '''
    Representation of a 0D cardiovascular model
    '''

    def __init__(self, vessels_df, parameters_array,
                 param_id_name_and_vals=None, param_id_date=None):
        '''
        Constructor
        '''
        self.vessels_df = vessels_df
        self.parameters_array = parameters_array
        self.param_id_date = param_id_date
        self.param_id_name_and_vals = param_id_name_and_vals
        self.possible_vessel_types = None
        self.possible_BC_types = None


    
    
        
