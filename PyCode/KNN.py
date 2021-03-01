
import json
import os
import re
import ast
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, make_scorer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import requests as reqs
# from pandas.io.json import json_normalize  

COLUMNS = ['specimen_num','citation_num', 'citation_code','specimen_code','igsn','archive_institution','specimen_material','taxon','taxon_rock_type', 'taxon_rock_class','rock_class_detail','rock_class_details','specimen_names','specimen_name','specimen_comments','specimen_comment','alteration','alterations','rock_textures','rock_texture','geological_ages','geological_ages__prefix','geological_ages__age','geological_ages__min','geological_ages__max','points_latitude','points_longitude','points_text','station_num', 'station_code','lat_label', 'long_label','points','centerLat','centerLong', 'elevation_min','elevation_max','location_precisions','geographic_location','tectonic_settings','expedition_num','expedition_code','sampling_technique_code','sampling_technique_name','analyzed_samples']
COLUMNS_COMPARE = ['citation_num','taxon_rock_type', 'taxon_rock_class','rock_class_details','specimen_name','points_latitude','points_longitude','elevation_min','elevation_max','geographic_location','tectonic_settings','expedition_num']
petdb_url = 'https://ecapi.earthchem.org/specimen'
# example_sample = '26639'
class KNN_builder:
    def __init__(self,pathtojson=['../json_files','../json_files_1']):
        self._files = np.array([])
        self._model = None
        for path in pathtojson:
            pet_path = os.path.join(os.path.dirname(__file__), path)
            for i in os.listdir(pet_path):
                if i.endswith('.json'):
                    full_path = '%s/%s' % (pet_path, i)
                    jsonStr = open(full_path, 'r', encoding='utf-8').read()
                    jsonStr = jsonStr.replace("'",'"')
                    jsonStr = jsonStr.replace('None','"None"')
                try:
                    jsonData = json.loads(jsonStr)
                    self._files = np.append(self._files,jsonData)
                except Exception as e:
                    print(e)
                    continue
    def _make_data(self):
        vfunc = np.vectorize(lambda x: x['data'] )
        stripped_data = vfunc(self._files)
        return pd.DataFrame.from_dict(pd.json_normalize(stripped_data), orient='columns')
    def format_data(self,df1=None):
       
        new_df = pd.DataFrame(data = None, index=None, columns = COLUMNS)
        df1 = self._make_data() if df1 is None else df1
        for index, row in df1.iterrows():
            citation_num_list = []
            citation_code_list = []
            specimen_num = row['specimen_num']
            specimen_code = row['specimen_code']
            igsn = row['igsn']
            archive_institution = row['archive_institution']
            specimen_material = row['specimen_material']
            taxon = row['taxon'] 
            rock_class_details = row['rock_class_details']
            if len(rock_class_details) == 0:
                rock_class_details = 'None'
            specimen_names = row['specimen_names']  
            specimen_comments = row['specimen_comments']
            if len(specimen_comments) == 0:
                specimen_comments = 'None'
            alterations = row['alterations']
            if len(alterations) == 0:
                alterations = 'None'
            rock_textures = row['rock_textures'] 
            geological_ages = row['geological_ages'] 
            station_num = row['station_num']
            station_code = row['station_code']
            lat_label = row['lat_label']
            long_label = row['long_label']
            points = row['points'] 
            latitude = ''
            longitude = ''
            text = ''
            if len(points)> 0:
                latitude = points[0]['latitude']
                longitude = points[0]['longitude']
                text = points[0]['text']
            centerLat = row['centerLat']
            centerLong = row['centerLong']
            elevation_min = row['elevation_min']
            elevation_max = row['elevation_max']
            location_precisions = row['location_precisions']
            geographic_location = row['geographic_location']
            tectonic_settings = row['tectonic_settings']
            expedition_num = row['expedition_num']
            expedition_code = row['expedition_code']
            sampling_technique_code = row['sampling_technique_code']
            sampling_technique_name = row['sampling_technique_name']
            analyzed_samples = row['analyzed_samples']
            taxon_rock_type = taxon[0]['rock_type']
            taxon_rock_class = taxon[0]['rock_class']
            
            
            geological_prefix = 'None'
            geological_ages__age = 'None'
            geological_ages__min = 'None'
            geological_ages__max = 'None'

            if len(geological_ages) > 0:
                geological_prefix = geological_ages[0]['prefix']
                geological_ages__age = geological_ages[0]['age']
                geological_ages__min = geological_ages[0]['age_min']
                geological_ages__max = geological_ages[0]['age_max']

            
            rock_texture = 'None'
            for item in rock_textures:
                rock_texture = item['rock_texture']
            
            rock_class_detail = 'None'
            if rock_class_details != "None":
                rock_class_detail = rock_class_details[0]['rock_class_detail']
            
            alteration = 'None'
            if alterations != "None":
                alteration = alterations[0]['alteration']
            
            
            specimen_comment = 'None'
            if specimen_comments != "None":
                specimen_comment = specimen_comments[0]['comment']
            
            
            
            specimen_names_list = [] 
            
            for item in specimen_names:
                specimen_name = item['specimen_name']
                specimen_names_list.append(specimen_name)

            for item in taxon:
                for source in item['source']:
                    citation_num = source['citation_num']
                    citation_code = source['citation_code']
                    citation_num_list.append(citation_num)
                    citation_code_list.append(citation_code)
            
            #print(specimen_names_list)
            
            i = 0
            while i < len(citation_num_list): #
                new_row = [] 
                citation_num = citation_num_list[i]
                citation_code = citation_code_list[i]
                
                #print('citation_num', citation_num)
                #print('citation_code', citation_code)
                #print(i)
                
                specimen_name = ""
                if i < len(specimen_names_list):
                    specimen_name = specimen_names_list[i]
                #print(specimen_name)
                
                new_row.append(specimen_num)   
                new_row.append(citation_num) 
                new_row.append(citation_code) 
                new_row.append(specimen_code)
                new_row.append(igsn)
                new_row.append(archive_institution)
                new_row.append(specimen_material)
                new_row.append(taxon)
                new_row.append(taxon_rock_type)
                new_row.append(taxon_rock_class)
                new_row.append(rock_class_detail)
                new_row.append(rock_class_details)
                new_row.append(specimen_names)
                new_row.append(specimen_name)
                new_row.append(specimen_comments)
                new_row.append(specimen_comment)
                new_row.append(alteration)
                new_row.append(alterations)
                new_row.append(rock_textures)
                new_row.append(rock_texture)
                new_row.append(geological_ages)
                new_row.append(geological_prefix)
                new_row.append( geological_ages__age)
                new_row.append( geological_ages__min)
                new_row.append(geological_ages__max)
                new_row.append(latitude)
                new_row.append(longitude)
                new_row.append(text)
                new_row.append(station_num)
                new_row.append(station_code)
                new_row.append(lat_label)
                new_row.append(long_label)
                new_row.append(points)
                new_row.append(centerLat)
                new_row.append(centerLong)
                new_row.append(elevation_min)
                new_row.append(elevation_max)
                new_row.append(location_precisions)
                new_row.append(geographic_location)
                new_row.append(tectonic_settings)
                new_row.append(expedition_num)
                new_row.append(expedition_code)
                new_row.append(sampling_technique_code)
                new_row.append(sampling_technique_name)
                new_row.append(analyzed_samples)
                
                
                a_series = pd.Series(new_row, index=new_df.columns) # 
                new_df = new_df.append(a_series, ignore_index=True) 
                i += 1       

        #         pd.set_option("display.max_columns", None)
        # pd.set_option("display.max_colwidth", None)
        #pd.set_option('display.max_rows', None)
        new_df.drop('taxon', axis = 1, inplace = True)
        new_df.drop('specimen_names', axis = 1, inplace = True)
        new_df.drop('rock_textures', axis = 1, inplace = True)
        new_df.drop('geological_ages', axis = 1, inplace = True)
        new_df.drop('points', axis = 1, inplace = True)
        new_df.drop('rock_class_details', axis = 1, inplace = True)
        new_df.drop('alterations', axis = 1, inplace = True)
        new_df.drop('specimen_comments', axis = 1, inplace = True)

        new_df['geographic_location'] = new_df['geographic_location'].apply(lambda x: ', '.join(map(str, x))) # convert list to strings
        new_df['tectonic_settings'] = new_df['tectonic_settings'].apply(lambda x: ', '.join(map(str, x)))
        new_df['location_precisions'] = new_df['location_precisions'].apply(lambda x: ', '.join(map(str, x)))
        new_df['analyzed_samples'] = new_df['analyzed_samples'].apply(lambda x: ', '.join(map(str, x)))
        #new_df['specimen_name'] = new_df['specimen_name'].apply(lambda x: ', '.join(map(str, x)))

        new_df = new_df.replace('', 'None')
        new_df = new_df.replace('N/A', 'None')
        new_df = new_df.rename(columns = {'alteration' : 'alterations', 'rock_class_detail' : 'rock_class_details', 'specimen_comment' : 'specimen_comments'})
        new_df_final = new_df.drop(columns=['alterations', 'analyzed_samples', 'archive_institution', 'citation_code', 'geological_ages__prefix', 'geological_ages__age', 'geological_ages__min', 'geological_ages__max','igsn','lat_label','long_label','location_precisions', 'rock_texture', 'specimen_comments','sampling_technique_code', 'sampling_technique_name', 'specimen_material', 'centerLat', 'centerLong', 'points_text'])
    
        return new_df_final
    """
    map and reduce verison *dig a little deeper into 
    """ 
    def count_similar_columns_map(self,row1,row2,columns=COLUMNS_COMPARE):
        comparisons = list(map(lambda column: row1[str(column)] == row2[str(column)],columns))
        return len(list(filter(lambda x: x,comparisons)))
    """
    comparing rows 
    """
    def compare_two_rows_map(self,row1,row2,columnLen,threshold=.50):
        if row1['specimen_num'] == row2['specimen_num']:
            counts = self.count_similar_columns_map(row1,row2)
            return float(counts/columnLen) >= threshold
        return False

    def compare_rows_against_df(self,df_whole):
        results = []
        for i, row1 in df_whole.iterrows():
            a = np.array([[ i, j,self.compare_two_rows_map(row1,row2,len(COLUMNS_COMPARE))] for j,row2 in df_whole.iterrows() if i != j ])
            results.append(a)
        results = np.array(results)
        shaperesults = results.shape
   
        reshaped_data = np.reshape(results,(shaperesults[0]*shaperesults[1],3))
 
        return pd.DataFrame({'row1':reshaped_data[:,0],'row2':reshaped_data[:,1],'isMatch':reshaped_data[:,2]})  

    def combine_similiaries(self,originalDF,similarDF):
        row1DF = originalDF.iloc[similarDF['row1']]
        row2DF = originalDF.iloc[similarDF['row2']]
        columns1 = ["{}_0".format(column) for column in row1DF.columns]
        columns2 = ["{}_1".format(column) for column in row2DF.columns]
        row1DF.columns = columns1
        row2DF.columns = columns2

        row1DF.reset_index(drop=True,inplace=True)
        row2DF.reset_index(drop=True,inplace=True)
  
        combine = pd.concat([row1DF,row2DF],axis=1)
        combine['isMatch'] = similarDF['isMatch']
        return combine

    def pre_processing(self):
        data = self.format_data()
        isMatchDF = self.compare_rows_against_df(data)
        combineDF = self.combine_similiaries(data,isMatchDF)
        doubledf1 = pd.get_dummies(combineDF)
        print(doubledf1)
        df_majority = doubledf1[doubledf1.isMatch==0]
        df_minority = doubledf1[doubledf1.isMatch==1]
        
        # Downsample majority class
        df_majority_downsampled = resample(df_majority, 
                                        replace=False,    # sample without replacement
                                        n_samples=len(df_minority), # to match minority class
                                        random_state=123) # reproducible results
        
        # Combine minority class with downsampled majority class
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])

        return df_downsampled
    def build_model(self):
        df_sample = self.pre_processing()
        # df_sample = df_sample.values
        X = df_sample.loc[:,df_sample.columns != 'isMatch'].values
        Y = df_sample['isMatch'].values

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.3, random_state=0)
        
        self._model = KNeighborsClassifier()
        self._model.fit(X_train, Y_train)
        print(X.shape)
        knn_pred = self._model.predict(X_test)
        knn_accuracy = accuracy_score(Y_test, knn_pred)
        print(knn_accuracy)
        return df_sample.columns
    
    def compare_two_sample(self,specimenA,specimenB):
        df_columns = []
        if self._model == None:
            df_columns=self.build_model()
            
        rqA = reqs.get(petdb_url+'/'+specimenA,verify=False)
        rqB =  reqs.get(petdb_url+'/'+specimenB,verify=False)

        DF_A = pd.DataFrame.from_dict(pd.json_normalize(rqA.json()['data']), orient='columns')
        DF_B = pd.DataFrame.from_dict(pd.json_normalize(rqB.json()['data']), orient='columns')
        DF_combine = pd.concat([DF_A,DF_B])
        DF_format = self.format_data(DF_combine)
        row1DF = DF_format.iloc[[0]]
        row2DF = DF_format.iloc[[1]]
    
        columns1 = ["{}_0".format(column) for column in row1DF.columns]
        columns2 = ["{}_1".format(column) for column in row2DF.columns]
        row1DF.columns = columns1
        row2DF.columns = columns2

        row1DF.reset_index(drop=True,inplace=True)
        row2DF.reset_index(drop=True,inplace=True)

        combine = pd.concat([row1DF,row2DF],axis=1)
        explode_df = pd.get_dummies(combine)
        padding =[]
        for col in df_columns:
            if col == 'isMatch':
                pass 
            elif col in explode_df.columns:
                padding.append(explode_df[col].values[0])
            else:
                padding.append(0)
        test_data = np.array(padding)
        test_data = test_data.reshape(1,-1)
        return self._model.predict_proba(test_data))


knn = KNN_builder()
test_prob = knn.compare_two_sample('26639','26639')


    