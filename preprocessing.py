import pandas as pd
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os


class clean_data():
    
    '''
    
    This is where data cleaning and engineering is done. The issues of NaN,
    outliers, undefined columns etc will be taken care of
    
    '''
    
    def __init__(self, data):
        
        self.data = data
        self.columns = self.data.columns.tolist()
        
        
    def identify_columns(self, high_dim=100):
        
        """
        
        This funtion takes in the data, identifies the numerical, low categorical,
        binary and high categorical attributes and stores them in a list
        
        """
        
        self.num_attributes = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_attributes = self.data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        self.low_cat = []
        self.hash_features = []
        self.binary_cat = []

        for item in self.cat_attributes:
            if self.data[item].nunique() > high_dim:
                print('\n {} has a high cardinality. It has {} unique attributes'
                      .format(item, self.data[item].nunique()))
                self.hash_features.append(item)
                
            elif self.data[item].nunique() < high_dim and self.data[item].nunique() > 2:
                print('\n {} has a relatively low cardinality. It has {} unique attributes'
                     .format(item, self.data[item].nunique()))
                self.low_cat.append(item)
                
            elif self.data[item].nunique() == 2:
                print('\n {} has a binary classification with {} unique attributes'.
                     format(item, self.data[item].nunique()))
                
            else:
                print('\n {} is a categorical variable with {} number of attributes'.
                     format(item, self.data[item].nunique()))
                
    def handle_mixed_dtypes(self):
        """
        This method handles the mixed data type found in certain columns
        input: original dataframe
        
        """
        mixed = ['CAMEO_DEUG_2015', 'CAMEO_INTL_2015']
        self.data[mixed] = self.data[mixed].replace({'X': np.nan, 'XX':np.nan})
        self.data[mixed] = self.data[mixed].astype(float)
        
    def handle_unknown(self):
        
        """
        This method replaces unknown values with NaN (0, -1, 9). The excluded list(to_pass)
        are the valid attributes with data point 9 (which is not missing)
        
        """
        self.identify_columns()
        self.to_pass = ['LP_FAMILIE_FEIN', 'LP_FAMILIE_GROB', 'LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB',
                  'LP_STATUS_FEIN', 'LP_STATUS_GROB', 'PRAEGENDE_JUGENDJAHRE', 'WOHNDAUER_2008',
                  'ORTSGR_KLS9', 'GFK_URLAUBERTYP', 'D19_VERSAND_ONLINE_QUOTE_12', 
                    'D19_VERSAND_ONLINE_DATUM', 'D19_VERSAND_OFFLINE_DATUM', 'D19_VERSAND_DATUM',
                    'D19_TELKO_ONLINE_DATUM', 'D19_TELKO_OFFLINE_DATUM', 'D19_TELKO_DATUM',
                  'D19_KONSUMTYP', 'D19_GESAMT_ONLINE_QUOTE_12', 'D19_GESAMT_ONLINE_DATUM', 
                   'D19_GESAMT_OFFLINE_DATUM', 'D19_GESAMT_DATUM', 'D19_BANKEN_ONLINE_QUOTE_12',
                   'D19_BANKEN_ONLINE_DATUM', 'D19_BANKEN_OFFLINE_DATUM', 'D19_BANKEN_DATUM',
                   'CAMEO_DEUG_2015','ALTER_HH', 'ALTERSKATEGORIE_GROB']
        
        self.difference = list(set(self.num_attributes) - set(self.to_pass))
        
        if 'RESPONSE' in self.difference:
            self.difference.remove('RESPONSE')
            
        self.data[self.difference] = self.data[self.difference].astype(float).replace({-1: np.nan, 0: np.nan, 9: np.nan})
        self.data[self.to_pass] = self.data[self.to_pass].astype(float).replace({-1: np.nan, 0: np.nan})
        
    
    def check_nan_before(self):
        
        """
        
        Function checks if NaN values are present in the dataset for both 
        categorical and numerical variables before handling any irregularities
    
        """
        missing_values = self.data.isnull().sum()
        count = missing_values[missing_values>1]
        print('\n Features       Count of missing value')
        print('{}'.format(count))
        return count
    
    
    def check_nan_after(self):
        
        """
        
        Function checks if NaN values are present in the dataset for both 
        categorical and numerical variables after handling irregularities
    
        """
        self.handle_mixed_dtypes()
        self.handle_unknown()
        missing_values = self.data.isnull().sum()
        count = missing_values[missing_values>1]
        print('\n Features       Count of missing value')
        print('{}'.format(count))
        return count
    
    def plot_nan(self, nan_values):
        
        """
        
        Here is where top 30 missing values are plotted before and after NaN
        values are detected using separate conditions
        
        """
        plot = nan_values.sort_values(ascending=False)[:30]
        
        # Figure Size 
        fig, ax = plt.subplots(figsize =(16, 9)) 

        # Horizontal Bar Plot 
        ax.barh(plot.index, plot.values) 

        # Remove axes splines 
        for s in ['top', 'bottom', 'left', 'right']: 
            ax.spines[s].set_visible(False) 
        # Remove x, y Ticks 
        ax.xaxis.set_ticks_position('none') 
        ax.yaxis.set_ticks_position('none') 

        # Add padding between axes and labels 
        ax.xaxis.set_tick_params(pad = 5) 
        ax.yaxis.set_tick_params(pad = 10) 

        # Add x, y gridlines 
        ax.grid(b = True, color ='grey', 
                linestyle ='-.', linewidth = 0.5, 
                alpha = 0.2) 

        # Show top values  
        ax.invert_yaxis() 

        # Add annotation to bars 
        for i in ax.patches: 
            plt.text(i.get_width()+0.2, i.get_y()+0.5,  
                     str(round((i.get_width()), 2)), 
                     fontsize = 10, fontweight ='bold', 
                     color ='grey') 
        # Add Plot Title 
        ax.set_title('Chart showing the top 30 missing values in the general population dataset', 
                     loc ='left', ) 

        # Add Text watermark 
        fig.text(0.9, 0.15, 'Alvaro', fontsize = 12, 
                 color ='grey', ha ='right', va ='bottom', 
                 alpha = 0.7) 

        # Show Plot 
        plt.show() 
            
        
    
    def drop_na(self, row=True, column=True):
        
        '''
        This function drops NaN values if they are up to 30% across the colums
        and up to 50 for the rows
        
        Args:
            row: to determe if the NaN values are to be dropped across rows
            columns: to determine if the NaN values are to be dropped across columns
        
        '''
        
        self.handle_mixed_dtypes()
        self.handle_unknown()
        
        if column:
            threshold = int(len(self.data) * 0.30)
            self.data = self.data.dropna(axis = 1, thresh = threshold)
            
        elif row:
            self.data = self.data.dropna(axis = 0, thresh = 50)
            
        else:
            pass
    

    
    def engineering(self):
        
        """
        
        This compiles previous functions together. In addition, some new attributes will
        be created and old attributes will be modified to help with all pre-processing
        and data engineering.
    
        """
        
        self.drop_na()
        #Breaking down the PRAEGENDE_JUGENDJAHRE attribute with movement and flag in mind
        demography_dictionary = {1: 40, 2: 40, 3: 50, 4: 50, 5: 60, 6: 60, 7: 60,
                        8: 70, 9: 70, 10: 80, 11: 80, 12: 80, 13: 80,
                        14: 90, 15: 90}
        self.data['PRAEGENDE_JUGENDJAHRE_DEMOGRAPHY'] = self.data['PRAEGENDE_JUGENDJAHRE'].map(demography_dictionary)
        
        reunification = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 1, 8: 0,
                         9: 1, 10: 0, 11: 1, 12: 0, 13: 1, 14: 0, 15: 1}
        self.data['PRAEGENDE_JUGENDJAHRE_REUNIFICATION'] = self.data['PRAEGENDE_JUGENDJAHRE'].map(reunification)
        
        #Handling the WOHNLAGE attribute  to indicate poor/not poor neighbourhood
        neighbourhood_dict = {1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0, 7.0: 1, 8.0: 1}
        self.data['WOHNLAGE_NEIGHBOURHOOD'] = self.data['WOHNLAGE'].map(neighbourhood_dict)
        
        #Mapping the TITEL_KZ attribute to indicate where the user has a title or not
#         title_dict = {1:1, 2:1, 3:1, 4:1, 5:0}
#         self.data['TITEL_KZ_TITLE'] = self.data['TITEL_KZ'].map(title_dict)
        
        #Dealing with the LP_LEBENSPHASE_FEIN attribute
        demography = {1: 'younger_age', 2: 'middle_age', 3: 'younger_age', 4: 'middle_age',
                      5: 'advanced_age', 6: 'retirement_age', 7: 'advanced_age',
                      8: 'retirement_age', 9: 'middle_age', 10: 'middle_age', 
                      11: 'advanced_age', 12: 'retirement_age',
                      13: 'advanced_age', 14: 'younger_age', 15: 'advanced_age',
                      16: 'advanced_age', 17: 'middle_age', 18: 'younger_age',
                      19: 'advanced_age', 20: 'advanced_age', 21: 'middle_age',
                      22: 'middle_age', 23: 'middle_age', 24: 'middle_age',
                      25: 'middle_age', 26: 'middle_age', 27: 'middle_age',
                      28: 'middle_age', 29: 'younger_age', 30: 'younger_age',
                      31: 'advanced_age', 32: 'advanced_age', 33: 'younger_age',
                      34: 'younger_age', 35: 'younger_age', 36: 'advanced_age',
                      37: 'advanced_age', 38: 'retirement_age', 39: 'middle_age',
                      40: 'retirement_age'}
        
        
        affluence = {1: 'low', 2: 'low', 3: 'average', 4: 'average', 5: 'low', 
                     6: 'low', 7: 'average', 8: 'average', 9: 'average', 
                     10: 'wealthy', 11: 'average', 12: 'average', 13: 'top', 
                     14: 'average', 15: 'low', 16: 'average', 17: 'average',
                     18: 'wealthy', 19: 'wealthy', 20: 'top', 21: 'low', 
                     22: 'average', 23: 'wealthy', 24: 'low', 25: 'average',
                     26: 'average', 27: 'average', 28: 'top', 29: 'low',
                     30: 'average', 31: 'low', 32: 'average', 33: 'average',
                     34: 'average', 35: 'top', 36: 'average', 37: 'average',
                     38: 'average', 39: 'top', 40: 'top'}
        
        self.data['LP_LEBENSPHASE_FEIN_DEMOGRAPHY'] = self.data['LP_LEBENSPHASE_FEIN'].map(demography)
        self.data['LP_LEBENSPHASE_FEIN_AFFLUENCE'] = self.data['LP_LEBENSPHASE_FEIN'].map(affluence)
        
        demography_dict = {'younger_age': 1, 'middle_age': 2, 'advanced_age': 3,
                     'retirement_age': 4}
        affluence_dict = {'low': 1, 'average': 2, 'wealthy': 3, 'top': 4}
        
        self.data['LP_LEBENSPHASE_FEIN_DEMOGRAPHY'] = self.data['LP_LEBENSPHASE_FEIN_DEMOGRAPHY'].map(demography_dict)
        self.data['LP_LEBENSPHASE_FEIN_AFFLUENCE'] = self.data['LP_LEBENSPHASE_FEIN_AFFLUENCE'].map(affluence_dict)
        
        german_flag_dict = {'W':0, 'O':1}
        self.data['OST_WEST_KZ'] = self.data['OST_WEST_KZ'].map(german_flag_dict)
        
        
        self.data = self.data.drop(['EINGEFUEGT_AM', 'PRAEGENDE_JUGENDJAHRE', 
                                    'LP_LEBENSPHASE_FEIN', 'MIN_GEBAEUDEJAHR',
                                    'D19_LETZTER_KAUF_BRANCHE'], axis = 1)
        
        
    
    def handle_nan(self,strategy='mean',fillna='mode'):
        
        """
        
        Function handles NaN values in a dataset for both categorical
        and numerical variables
    
        Args:
            strategy: Method of filling numerical features
            fillna: Method of filling categorical features
        """
        self.engineering()
        self.identify_columns()
        
        if strategy=='mean':
            for item in self.data[self.num_attributes]:
                self.data[item] = self.data[item].fillna(self.data[item].mean())
        if fillna == 'mode':
            for item in self.data[self.cat_attributes]:
                self.data[item] = self.data[item].fillna(self.data[item].value_counts().index[0])
        else:
            for item in self.data[self.num_attributes]:
                self.data[item] = self.data[item].fillna(fillna)
                
        return self.data