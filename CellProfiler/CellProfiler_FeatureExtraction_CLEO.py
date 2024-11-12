import polars as pl
from sqlalchemy import create_engine
import re
import gc
import numpy as np
import pandas as pd
import os
import sys
import time
#from sqlalchemy import create_engine
import re
import matplotlib.pyplot as plt
import gc
#from pycytominer.operations.correlation_threshold import correlation_threshold
import polars as pl
import pandas as pd
# import plotly.figure_factory as ff
# import plotly.subplots as sp
# import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
from typing import Union, Literal, Tuple, Set, List, Dict
# from cytominer_eval import evaluate
import scanpy as sc
import anndata as ad
db_uri = 'postgresql://pharmbio_readonly:readonly@imagedb-pg-postgresql.services.svc.cluster.local/imagedb'

class CellProfilerData:
    def __init__(self, project, analysis_id_feat, acq_id,barcode,z ):
        self.project = project
        self.analysis_id_feat = analysis_id_feat
        self.acq_id = acq_id
        self.df = None
        self.df_qc = None
        self.df_meta = None
        self.dataframes = None
        self.locations = None
        self.barcode = barcode
        self.z =z
    def get_data_from_server(self,type='feature'):
        if type == 'feature':
            type = 'cp-features'
        elif type == 'qc':
            type = 'cp-qc'
        else:
            raise ValueError("Type must be either 'feature' or 'qc'")
        query = f"""
            SELECT *
            FROM image_analyses_per_plate
        WHERE project LIKE '{self.project}%%'
        AND meta->>'type' = '{type}'
        AND analysis_date IS NOT NULL
        ORDER BY plate_barcode 
        """
        engine = create_engine(db_uri)
        connection = engine.connect()
        self.df = pl.read_database(query, connection)
        connection.close()
        return self.df

    def check_duplicate_analysis(self):
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call get_data_from_server() first.")

        df = self.df.filter(pl.col('plate_acq_id') == self.acq_id)
        print(df)
        df_with_count = df.group_by('plate_acq_id').agg(pl.count().alias('size'))
        print(df.select('plate_acq_id'))
        df_dupes = df_with_count.filter(pl.col('size') > 1)
        if df_dupes.is_empty():
            print("OK, no duplicate results found")
            self.df = df
        else:
            #print("WARNING! Duplicate results found")
            #print(self.df.select('analysis_id'))
            self.df = self.df.filter(pl.col('analysis_id') == self.analysis_id_feat)

            
            #print(self.df.select('analysis_id'))

    def load_feature_data(self,featureFileNames=['featICF_nuclei', 'featICF_cells', 'featICF_cytoplasm']):
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call get_data_from_server() first.")
        print(self.df.select('results'))
        #featureFileNames = ['featICF_nuclei', 'featICF_cells', 'featICF_cytoplasm']
        print(self.df)
        dataframes = {}
        for i in featureFileNames:
            base = self.df.select('results').to_series()[0]
            file = base + i + '.parquet'
            dataframe = pl.read_parquet(file)
            new_columns = [str(col) + '_' + re.sub('_.*', '', re.sub('featICF_', '', i)) for col in dataframe.columns]
            dataframe = dataframe.rename({old: new for old, new in zip(dataframe.columns, new_columns)})
            dataframes[i] = dataframe
        self.dataframes = dataframes

    def merge_dataframes(self,featureFileNames=['featICF_nuclei', 'featICF_cells', 'featICF_cytoplasm'],CLEO=False):
        if CLEO:
            nuclei = 'nucleus'
        else:
            nuclei = 'nuclei'
        if self.dataframes is None:
            raise ValueError("Feature data is not loaded. Call load_feature_data() first.")
        
        df = self.dataframes[featureFileNames[0]]
        print('size', df.shape)
        
        df = df.join(
            self.dataframes[featureFileNames[1]],
            left_on=['Metadata_Barcode_'+nuclei, 'Metadata_Site_'+nuclei, 'Metadata_Well_'+nuclei, 'Parent_cells_'+nuclei],
            right_on=['Metadata_Barcode_cells', 'Metadata_Site_cells', 'Metadata_Well_cells', 'ObjectNumber_cells'],
            how='left'
        )
        print('size', df.shape)
        df = df.join(
            self.dataframes[featureFileNames[2]],
            left_on=['Metadata_Barcode_'+nuclei, 'Metadata_Site_'+nuclei, 'Metadata_Well_'+nuclei, 'Parent_cells_'+nuclei],
            right_on=['Metadata_Barcode_cytoplasm', 'Metadata_Site_cytoplasm', 'Metadata_Well_cytoplasm', 'ObjectNumber_cytoplasm'],
            how='left'
        )
        print('size', df.shape)
        df = df.drop_nulls()
        print('size', df.shape)
        del self.dataframes
        gc.collect()
        df = df.with_columns([
            pl.lit(self.z).alias('Z')
        ])
        df = df.with_columns(pl.col('Metadata_Barcode_'+nuclei).str.replace("65", "64"))
        self.df = df


    def split_feat_metadata(self,CLEO=True):
        def is_meta_column(c):
            for ex in '''
                ^[a-z]
                Metadata
                ^Count
                ImageNumber
                Object
                Parent
                Children
                Plate
                Well
                Location
                _[XYZ]_
                _[XYZ]$
                BoundingBox
                Phase
                Orientation
                Angle
                Scale
                Scaling
                Width
                Height
                Group
                FileName
                PathName
                URL
                Execution
                ModuleError
                LargeBrightArtefact
                MD5Digest
                _ER
                ER_
                _MITO
                MITO_
                '''.split():
                if re.search(ex, c):
                    return True
            return False

        if self.df is None:
            raise ValueError("DataFrame is not merged. Call merge_dataframes() first.")
        if CLEO:
            nuclei = 'nucleus'
        else:
            nuclei = 'nuclei'
        self.df = self.df.rename({
        'Metadata_Barcode_'+nuclei: 'Metadata_Plate',
        'Metadata_Well_'+nuclei: 'Metadata_Well',
        'Metadata_Site_'+nuclei: 'Metadata_Site'
        })
        #self.df = self.df.with_columns(pl.col("Metadata_Plate").str.replace("65", "64"))
        #print(self.df)
        self.meta = self.df.select('Metadata_Plate', 'Metadata_Well', 'Metadata_Site','ObjectNumber_'+nuclei).with_row_index()
        self.locations = self.df
        self.meta = self.meta.with_columns([
            pl.lit(self.z).alias('Z')
        ])
        feat = self.df.select([col for col in self.df.columns
                              if '[Float64]' in str(self.df.select(col).dtypes) or '[Float32]' in str(self.df.select(col).dtypes) or 'int' in str(self.df[col].dtype)
                              if not is_meta_column(col)])
                        
        #self.df = self.df.select(feat.columns,'Metadata_Well')
        self.df = self.df.select(feat.columns).with_row_index()
        
    def load_compound(self,barcode,csv_file=False):
        if csv_file:
            compound = pl.read_csv(csv_file)
            
            compound = compound.rename({'Metadata_barcode': 'barcode'})
            
            
            compound = compound.filter(pl.col('barcode') == barcode )
            
            compound
            # rename colum batch_id to batch_nr
            compound = compound.rename({'batch_id': 'batch_id'})
            compound = compound.rename({'well_position': 'well_id'})
            
    
        else:
            db_uri = 'postgresql://pharmbio_readonly:readonly@imagedb-pg-postgresql.services.svc.cluster.local/imagedb'
            query = f"""
                SELECT *
                FROM plate_v1
                WHERE barcode LIKE '%%{barcode}%%'
                """

            # Query database and store result in pandas dataframe
            print("Select table with database...please wait")
            compound = pd.read_sql_query(query, db_uri)
            #drop duplicates
            compound = compound.drop_duplicates(subset=['well_id'])
            compound = pl.DataFrame(compound)
            
        self.meta = self.meta.join(compound.select('well_id','batch_id'), left_on='Metadata_Well',right_on='well_id', how='left')
        return compound
        

    def aggregate_dataframe(self, function='median'):
        if self.df is None:
            raise ValueError("DataFrame is not merged. Call merge_dataframes() first.")
        
        if function == 'median':
            df = self.df.group_by(['Metadata_Barcode_nucleus', 'Metadata_Site_nucleus', 'Metadata_Well_nucleus']).agg(
        
                [pl.col(col).median().alias(col) for col in self.df.columns if col not in ['Metadata_Barcode_nucleus', 'Metadata_Site_nucleus', 'Metadata_Well_nucleus']]
            )
        elif function == 'mean':
            df = self.df.group_by(['Metadata_Barcode_nuclei', 'Metadata_Site_nuclei', 'Metadata_Well_nuclei']).agg(
                [pl.col(col).mean().alias(col) for col in self.df.columns if col not in ['Metadata_Barcode_nuclei', 'Metadata_Site_nuclei', 'Metadata_Well_nuclei']]
            )
        self.df = df

    #blocklist_features = [col for col in normalized_profiles.columns if "Correlation_Manders" in col and "_nuclei" in col] +[col for col in normalized_profiles.columns if "Correlation_RWC" in col and "_nuclei" in col] +[col for col in normalized_profiles.columns if "Granularity_14" in col and "_nuclei" in col] + [col for col in normalized_profiles.columns if "Granularity_15" in col and "_nuclei" in col] +[col for col in normalized_profiles.columns if "Granularity_16" in col and "_nuclei" in col]
    #features = [feat for feat in normalized_profiles_merge.columns if feat not in meta_features and feat not in blocklist_features and feat not in meta_df_features]
    def normalize_MAD(self):
        #def drop_corr(df, threshold=0.9, samples='all'):
            #from pycytominer.operations.correlation_threshold import correlation_threshold
            # exclude = correlation_threshold(
            #     df,
            #     features=list(df.columns),
            #     threshold=threshold,
            #     samples=samples,
            # )
            #print(f'Dropping {len(exclude)=} columns (starting from {df.shape=})')
            #return df.map_data(lambda df: df.drop(columns=exclude))
            #return df.drop(exclude, axis=1)
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call merge_dataframes() first.")
        if self.meta is None:
            raise ValueError("Metadata DataFrame is not loaded.")
        meta = self.meta.to_pandas()
        feat = self.df.to_pandas()
        blocklist_features = [col for col in feat.columns if "Correlation_Manders" in col and "_nuclei" in col] +[col for col in feat.columns if "Correlation_RWC" in col and "_nuclei" in col] +[col for col in feat.columns if "Granularity_14" in col and "_nuclei" in col] + [col for col in feat.columns if "Granularity_15" in col and "_nuclei" in col] +[col for col in feat.columns if "Granularity_16" in col and "_nuclei" in col]
        feat = feat.drop(blocklist_features, axis=1)

        #feat = feat.loc[:, (feat.std() > 0.001) ]
        #feat = drop_corr(feat, threshold=0.9)
        # Filter self.meta based on batch_nr == 'PB001'
        df_dmso = feat[meta['batch_id'] == 'PHB000001']
        # Extract numeric columns for normalization (excluding Metadata_Well)
        
        # Calculate median and MAD for normalization
        df_dmso_median = df_dmso.median()
        df_dmso_MAD = (df_dmso - df_dmso_median).abs().median()
        
        # Perform normalization operation
        feat = (feat- df_dmso_median) / df_dmso_MAD
        feat = feat.dropna(axis=1)
        lower = feat.quantile(0.01)
        upper = feat.quantile(0.99)
        feat = feat.clip(lower=-10, upper=10)
        self.df = pl.DataFrame(feat)
        '''
    def Zmean(self):
        
        df_DMSO_meta = self.meta.filter(pl.col('batch_id') == 'PHB000001')
        df_DMSO_index = df_DMSO_meta.select('index')
        df_DMSO_feat = self.df.join(df_DMSO_index, on='index', how='inner')

        float_columns = [col for col in df_DMSO_feat.columns if col != 'index' or col !='AreaShape_Area_nucleus']
        float_columns = float_columns
        #df_DMSO = self.df.select(float_columns)
        df_DMSO = df_DMSO_feat.select(float_columns)

        mu = df_DMSO.select(float_columns).mean()
        std = df_DMSO.select(float_columns).std()
        #df = self.df
        print(mu)
        print(std)
        print(float_columns)
        
        for col in mu.columns:
            if mu[col].is_null().any():
                raise RuntimeError(f"some mean value in column {col} is nan?!")
            if mu[col].is_infinite().any():
                raise RuntimeError(f"some mean value in column {col} is infinite?!")
        std = std.select([pl.col(c).map_dict({0: 1}, default=pl.col(c)) for c in std.columns])
        self.df = self.df.with_columns([(pl.col(c) - mu[c]) / (std[c]+0.01) for c in mu.columns])
        # make all columns float 32
        for col in self.df.columns:
            self.df = self.df.with_columns(pl.col(col).cast(pl.Float32))
        '''
    def Zmean(self):
        #print(self.df)
        #print(self.meta)
        if 'batch_id' in self.meta.columns:
            df_DMSO_meta = self.meta.filter(pl.col('batch_id') == 'PHB000001')
        else:
            df_DMSO_meta = self.meta.filter(pl.col('batch-id') == 'PHB000001')
        df_DMSO_index = df_DMSO_meta.select('index')
        df_DMSO_feat = self.df.join(df_DMSO_index, on='index', how='inner')

        float_columns = [col for col in df_DMSO_feat.columns if col != 'index' ]
        #float_columns = [col for col in float_columns if col != 'AreaShape_Area_nucleus' ]
        

        #df_DMSO = self.df.select(float_columns)
        df_DMSO = df_DMSO_feat.select(float_columns)
        #print(df_DMSO)
        mu = df_DMSO.select(float_columns).mean()
        std = df_DMSO.select(float_columns).std()
        
        #df = self.df
        for col in mu.columns:
            if mu[col].is_null().any():
                raise RuntimeError(f"some mean value in column {col} is nan?!")
            if mu[col].is_infinite().any():
                raise RuntimeError(f"some mean value in column {col} is infinite?!")
                   ### OLD PYTHON
        #std = std.select([pl.col(c).map_dict({0: 1}, default=pl.col(c)) for c in std.columns])
        #self.df = self.df.with_columns([(pl.col(c) - mu[c]) / (std[c]+0.01) for c in mu.columns])
        std = std.select([
        pl.when(pl.col(c) == 0).then(1).otherwise(pl.col(c)).alias(c) for c in std.columns
        ])
        self.df = self.df.with_columns([(pl.col(c) - mu[c]) / (std[c] + 0.01) for c in mu.columns])
        # make all columns float 32
        for col in self.df.columns:
            self.df = self.df.with_columns(pl.col(col).cast(pl.Float32))
        # replace 0 with 1 (specifically not clip) to avoid div by zero
    def Zmad(self,use_clipping=True):
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call merge_dataframes() first.")
        if self.meta is None:
            raise ValueError("Metadata DataFrame is not loaded.")
        # if use_clipping:
        #     lower_quantile = self.df.quantile(0.01)
        #     upper_quantile = self.df.quantile(0.99)
        #     print("calced quantiles")

        #     for col in self.df.columns:
        #         if col != 'index': 
        #             self.df = self.df.with_columns(pl.col(col).clip(lower=lower_quantile[col],upper=upper_quantile[col]))
        
        df_DMSO_meta = self.meta.filter(pl.col('batch_id') == 'PHB000001')
        df_DMSO_index = df_DMSO_meta.select('index')
        df_DMSO_feat = self.df.join(df_DMSO_index, on='index', how='inner')

        float_columns = [col for col in df_DMSO_feat.columns if col != 'index']
        
        df_DMSO = df_DMSO_feat.select(float_columns)
        median = df_DMSO.select(float_columns).median()
        mad = df_DMSO.select([(pl.col(c) - pl.col(c).median()).abs().alias(c) for c in float_columns])
        mad = pl.concat([self.df.select((pl.col(c)-pl.col(c).median()).abs().median()) for c in self.df.select(float_columns).columns], how='horizontal')
        
        #mad = mad.select([pl.col(c).map_dict({0: 0.01}, default=pl.col(c)) for c in mad.columns])
        mad = mad.select([pl.when(pl.col(c) == 0).then(0.01).otherwise(pl.col(c)).alias(c) for c in mad.columns])

        df_standardized = self.df.with_columns([(pl.col(c) - median[c]) / (mad[c]) for c in median.columns])
        
        # remove columns with all zeros
        
        # Check for null or infinite medians and raise errors if found
        for col in median.columns:
            if median[col].is_null().any():
                raise RuntimeError(f"some median value in column {col} is nan?!")
            if median[col].is_infinite().any():
                raise RuntimeError(f"some median value in column {col} is infinite?!")
        for i,col in enumerate(median.columns):
            if df_standardized[col].is_null().any():
                found_nan=True
                print(f"some value in column {col,i} is nan")
        for i, col in enumerate(mad.columns):
            if mad[col].is_null().any():
                raise RuntimeError(f"some MAD value in column {col,i} is nan?!")
            if mad[col].is_infinite().any():
                raise RuntimeError(f"some MAD value in column {col,i} is infinite?!")
            if (mad[col] == 0).any():
                raise RuntimeError(f"unexpected 0 in column {col}")
        #df_standardized = df_standardized.select([c for c in df_standardized.columns if df_standardized[c].sum() != 0])
        self.df = df_standardized
    def Zmad(self):
        # Check for 'batch_id' column in meta and filter DMSO records
        if 'batch_id' in self.meta.columns:
            df_DMSO_meta = self.meta.filter(pl.col('batch_id') == 'PHB000001')
        else:
            df_DMSO_meta = self.meta.filter(pl.col('batch-id') == 'PHB000001')
        df_DMSO_index = df_DMSO_meta.select('index')
        df_DMSO_feat = self.df.join(df_DMSO_index, on='index', how='inner')
    
        # Identify float columns, excluding 'index'
        float_columns = [col for col in df_DMSO_feat.columns if col != 'index']
    
        # Select only float columns from DMSO subset
        df_DMSO = df_DMSO_feat.select(float_columns)
        
        # Calculate the median and MAD for each column
        median = df_DMSO.select(float_columns).median()
        mad = df_DMSO.select([(pl.col(c) - pl.col(c).median()).abs().median().alias(c) for c in float_columns])
    
        # Handle zero MAD values by replacing them with a small constant (0.01)
        mad = mad.select([
            pl.when(pl.col(c) == 0).then(0.000001).otherwise(pl.col(c)).alias(c) for c in mad.columns
        ])
    
        # Standardize columns in self.df using median and MAD
        self.df = self.df.with_columns([(pl.col(c) - median[c]) / (mad[c]) for c in median.columns])
    
        # Ensure all columns are cast to Float32
        for col in self.df.columns:
            self.df = self.df.with_columns(pl.col(col).cast(pl.Float32))
    def normalize_standard(self):
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call merge_dataframes() first.")
        if self.meta is None:
            raise ValueError("Metadata DataFrame is not loaded.")
            
        meta = self.meta.to_pandas()
        feat = self.df.to_pandas()
        
        # Filter features with standard deviation greater than 0.001
        feat = feat.loc[:, (feat.std() > 0.001)]

        # Filter self.meta based on batch_nr == 'PHB000001'
        df_dmso = feat[meta['batch_id'] == 'PHB000001']

        # Calculate mean and standard deviation for normalization
        df_dmso_mean = df_dmso.mean()
        df_dmso_std = df_dmso.std()

        # Perform normalization operation
        feat = (feat - df_dmso_mean) / df_dmso_std
        feat = feat.dropna(axis=1)
        #feat = feat.clip(lower=-10, upper=10)
        
        self.df = pl.DataFrame(feat)

# def compute_grit(df,meta):
#         #df = df.to_pandas()
#         #meta = meta.to_pandas()
#         meta['comp_replicate'] = meta['Metadata_Plate'] + '_' + meta['Metadata_Well'] + '_' + meta['Z'].astype(str) + '_' + meta['Metadata_Site'].astype(str)
#         df_merged = df.merge(meta, left_index=True, right_index=True)

#         one_plate = meta.Metadata_Plate.unique()
#         grit_scores = evaluate(
#             profiles=df_merged,  
#             features=list(df.columns),
#             #meta_features=list(dfZscores.columns[-7:]),# adjust after above
#             meta_features=meta.columns,   
#             replicate_groups={"profile_col": "comp_replicate", "replicate_group_col": "Batch_nr" },
#             operation="grit",
#             similarity_metric="pearson",
#             grit_replicate_summary_method="mean", # median
#             grit_control_perts=df_merged.query("Batch_nr == 'PHB000001'").comp_replicate.unique().tolist()
#         ).assign(one_plate=one_plate[0])
#         return grit_scores

# def drop_corr(df, threshold=0.9, samples='all'):
#             from pycytominer.operations.correlation_threshold import correlation_threshold
#             exclude = correlation_threshold(
#                 df,
#                 features=list(df.columns),
#                 threshold=threshold,
#                 samples=samples,
#             )
#             print(f'Dropping {len(exclude)=} columns (starting from {df.shape=})')
#             #return df.map_data(lambda df: df.drop(columns=exclude))
#             return df.drop(exclude, axis=1)

def pca_umap(adata):
    #sc.pp.scale(adata)
    sc.tl.pca(adata)
    sc.pl.pca_variance_ratio(adata, log=True)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    sc.tl.umap(adata)
    sc.pl.umap(adata)
    return adata

# import plotly.offline as pyo

# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.express as px
# import colorsys
'''
def generate_distinct_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return ['rgb' + str(tuple(int(255*x) for x in rgb)) for rgb in RGB_tuples]

def plot_umap(adata):
    umap_df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
    umap_df['Batch_nr'] = adata.obs['Batch_nr'].values
    umap_df['Metadata_Site'] = adata.obs['Metadata_Site'].values
    umap_df['Metadata_Well'] = adata.obs['Metadata_Well'].values
    umap_df['Z'] = adata.obs['Z'].values

    # Generate a color palette
    unique_batches = umap_df['Batch_nr'].unique()
    n_colors = len(unique_batches)
    
    # Combine multiple color palettes
    color_palette = (px.colors.qualitative.Plotly + 
                     px.colors.qualitative.Set1 + 
                     px.colors.qualitative.Set2 + 
                     px.colors.qualitative.Set3 + 
                     px.colors.qualitative.Pastel1 + 
                     px.colors.qualitative.Pastel2)
    
    # If we still need more colors, generate them
    if n_colors > len(color_palette):
        additional_colors = generate_distinct_colors(n_colors - len(color_palette))
        color_palette.extend(additional_colors)
    
    # Ensure we have exactly n_colors
    color_palette = color_palette[:n_colors]

    fig = go.Figure()

    # Always present trace for 'PHB000001'
    batch_phb0001_df = umap_df[umap_df['Batch_nr'] == 'PHB000001']
    fig.add_trace(go.Scatter(
        x=batch_phb0001_df['UMAP1'],
        y=batch_phb0001_df['UMAP2'],
        mode='markers',
        marker=dict(size=4, color='lightgray'),  # Reduced size
        name='PHB000001',
        text=[f"site: {site}, well: {well}, z: {z}" for site, well, z in 
              zip(batch_phb0001_df['Metadata_Site'], batch_phb0001_df['Metadata_Well'], batch_phb0001_df['Z'])],
        hoverinfo='text'
    ))

    # Add traces for other batches
    for i, batch in enumerate(unique_batches):
        if batch != 'PHB000001':
            batch_df = umap_df[umap_df['Batch_nr'] == batch]
            fig.add_trace(go.Scatter(
                x=batch_df['UMAP1'],
                y=batch_df['UMAP2'],
                mode='markers',
                marker=dict(size=7, color=color_palette[i]),  # Color from palette
                name=str(batch),
                text=[f"site: {site}, well: {well}, z: {z}, cmpd: {cmpd}" for site, well, z, cmpd in 
                      zip(batch_df['Metadata_Site'], batch_df['Metadata_Well'], batch_df['Z'], batch_df['Batch_nr'])],
                hoverinfo='text'
            ))

    # Create buttons for each batch to update the visibility
    buttons = []
    for batch in unique_batches:
        if batch != 'PHB000001':
            buttons.append(dict(
                method='update',
                label=str(batch),
                args=[{
                    'visible': [trace.name == str(batch) or trace.name == 'PHB000001' for trace in fig.data]
                }]
            ))

    # Add a button to show all batches
    buttons.append(dict(
        method='update',
        label='All',
        args=[{'visible': [True] * len(fig.data)}]
    ))

    # Update layout
    fig.update_layout(
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
        }],
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        title='UMAP Projection',
        width=1000,
        height=1000
    )

    return fig
    '''

# def generate_distinct_colors(n):
#     HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
#     RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
#     return ['rgb' + str(tuple(int(255*x) for x in rgb)) for rgb in RGB_tuples]

# def plot_umap(adata):
#     umap_df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
#     umap_df['Batch_nr'] = adata.obs['Batch_nr'].values
#     umap_df['Metadata_Site'] = adata.obs['Metadata_Site'].values
#     umap_df['Metadata_Well'] = adata.obs['Metadata_Well'].values
#     umap_df['Z'] = adata.obs['Z'].values

#     # Generate a color palette
#     unique_batches = umap_df['Batch_nr'].unique()
#     n_colors = len(unique_batches)
    
#     # Combine multiple color palettes
#     color_palette = (px.colors.qualitative.Plotly + 
#                      px.colors.qualitative.Set1 + 
#                      px.colors.qualitative.Set2 + 
#                      px.colors.qualitative.Set3 + 
#                      px.colors.qualitative.Pastel1 + 
#                      px.colors.qualitative.Pastel2)
    
#     # If we still need more colors, generate them
#     if n_colors > len(color_palette):
#         additional_colors = generate_distinct_colors(n_colors - len(color_palette))
#         color_palette.extend(additional_colors)
    
#     # Ensure we have exactly n_colors
#     color_palette = color_palette[:n_colors]

#     fig = go.Figure()

#     # Always present trace for 'PHB000001'
#     batch_phb0001_df = umap_df[umap_df['Batch_nr'] == 'PHB000001']
#     fig.add_trace(go.Scatter(
#         x=batch_phb0001_df['UMAP1'],
#         y=batch_phb0001_df['UMAP2'],
#         mode='markers',
#         marker=dict(size=4, color='lightgray'),  # Reduced size
#         name='Dimethyl Sulfoxide',
#         text=[f"site: {site}, well: {well}, z: {z}" for site, well, z in 
#               zip(batch_phb0001_df['Metadata_Site'], batch_phb0001_df['Metadata_Well'], batch_phb0001_df['Z'])],
#         hoverinfo='text'
#     ))

#     # Add traces for other batches
#     for i, batch in enumerate(unique_batches):
#         if batch != 'PHB000001':
#             batch_df = umap_df[umap_df['Batch_nr'] == batch]
#             fig.add_trace(go.Scatter(
#                 x=batch_df['UMAP1'],
#                 y=batch_df['UMAP2'],
#                 mode='markers',
#                 marker=dict(size=10, color=color_palette[i]),  # Color from palette
#                 name=str(batch),
#                 text=[f"site: {site}, well: {well}, z: {z}, cmpd: {cmpd}" for site, well, z, cmpd in 
#                       zip(batch_df['Metadata_Site'], batch_df['Metadata_Well'], batch_df['Z'], batch_df['Batch_nr'])],
#                 hoverinfo='text'
#             ))

#     # Create buttons for each batch to update the visibility
#     buttons = []
#     for batch in unique_batches:
#         if batch != 'PHB000001':
#             buttons.append(dict(
#                 method='update',
#                 label=str(batch),
#                 args=[{
#                     'visible': [trace.name == str(batch) or trace.name == 'Dimethyl Sulfoxide' for trace in fig.data]
#                 }]
#             ))

#     # Add a button to show all batches
#     buttons.append(dict(
#         method='update',
#         label='All',
#         args=[{'visible': [True] * len(fig.data)}]
#     ))

#     # Update layout
#     fig.update_layout(
#         updatemenus=[{
#             'buttons': buttons,
#             'direction': 'down',
#             'showactive': True,
#         }],
#         plot_bgcolor='white',
#         xaxis=dict(
#             showgrid=False,
#             zeroline=False,
#             showticklabels=False
#         ),
#         yaxis=dict(
#             showgrid=False,
#             zeroline=False,
#             showticklabels=False
#         ),
#         title='UMAP Projection',
#         width=1000,
#         height=1000
#     )

#     return fig

#     #filename = 'Umap_grit1.html'
#     #pyo.plot(fig, filename=filename, auto_open=False)
def merge_df_qc (df,df_qc):
    #display(df_qc)
    #display(df)
    df_merged = pd.merge(df, df_qc, left_on=['Metadata_Barcode_nucleus','Metadata_Well_nucleus','Metadata_Site_nucleus','Z'],right_on=['Metadata_Barcode','Metadata_Well','Metadata_Site','Z'], how='left',suffixes=('', '_qc'))    
    #locations = pd.merge(locations, df_qc, left_on=['Metadata_Well','Metadata_Site','Z'],right_on=['Metadata_Well','Metadata_Site','Z'], how='left',suffixes=('', '_qc'))
    df_merged = df_merged.loc[:,~df_merged.columns.duplicated()]
    #locations = locations.loc[:,~locations.columns.duplicated()]
    df_remove_flagged = df_merged[df_merged['Total'] == 0 ]
    #locations = locations[locations['Total'] == 0]
    print("Reduction by", (len(df_merged))-(len(df_remove_flagged)) )
    print("Number of flagged instances in QC was", len(df_qc[df_qc['Total'] == 1]))
    #columns_to_drop = ['Outlier', 'Total']
    #feature_columns = [fc for fc in df_remove_flagged.columns if all(exclude not in fc for exclude in columns_to_drop)]
    #df_remove_flagged = df_remove_flagged[feature_columns]
    df = pl.from_pandas(df_remove_flagged.drop(columns=df_qc.columns))
    df_qc = pl.from_pandas(df_remove_flagged.drop(columns=df.columns))
    #df_indices = df.select(pl.col("index")) 
    #locations = locations.filter(pl.col("index").is_in(df_indices))

    # display(df)
    #display(df_qc)
    return df