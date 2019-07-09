#!/usr/bin/env python3

############# NOTES #############
'''
Modified version of the pca weighting thing
'''

############# IMPORTS #############

import pandas as pd, numpy as np
import os, datetime, time
import requests
from io import BytesIO
from pprint import pprint
from pathlib import Path
from helper_methods import clean_country_name
from pca_weighting import PCAWeighting

############# METHODS #############


def aggregate_zones(inframe, zonelist, ctry_nm, collist, country_field = 'Country'):
	zones = inframe[inframe[country_field].isin(zonelist)]
	zones[country_field] = ctry_nm
	zones = zones[[country_field] + collist]
	zones = pd.DataFrame(zones.groupby([country_field]).mean())
	zones.reset_index(inplace = True)
	inframe = inframe[~inframe[country_field].isin(zonelist)]
	outframe = pd.concat([inframe, zones])
	outframe.reset_index(inplace =  True, drop = True)
	outframe.sort_values(inplace = True, by = country_field)
	return outframe


def get_world_bank_dataset(url, name, outpath = None):
	if outpath == None:
		outpath = os.getcwd()
	r = requests.get(url)
	p = BytesIO(r.content)
	df_list = pd.read_excel(p, sheet_name = None)
	df = df_list['Data']
	if name.count('.xls') == 0:
		name += '.xlsx'
	path = os.path.join(outpath, name)
	df.to_excel(path)


def clean_year_cols(colname):
	try:
		return str(int(colname))
	except:
		return colname


def get_years(inframe):
	curr_yr = int(datetime.datetime.now().year)
	yr_span = [str(x) for x in range(curr_yr - 5, curr_yr)]
	yr_nans = [float(float(inframe[str(x)].isnull().sum())/float(len(inframe))) for x in yr_span]
	if yr_nans[-1] > 0.9:
		yr_span = [str(int(x)-1) for x in yr_span]
	main_cols = ['Country'] + yr_span
	return main_cols


def prep_wb_data(inframe, ctry_list):
	inframe = inframe[~inframe[inframe.columns[0]].isin([np.nan, 'Last Updated Date', 0, 1])]

	inframe.reset_index(inplace = True, drop = True)
	collist = inframe.loc[0]; inframe = inframe.loc[1:]
	inframe.columns = [clean_year_cols(x) for x in collist]
	inframe.rename(inplace = True, columns = {'Country Name': 'Country'})
	# print(inframe.columns)
	inframe['Country'] = inframe['Country'].apply((lambda x : clean_country_name(x)))
	inframe = inframe[get_years(inframe)]
	inframe = inframe[inframe['Country'].isin(ctry_list)]
	inframe.reset_index(inplace = True, drop = True)
	return inframe


if __name__ == "__main__":

	### ----- NOTES ----- ###
	'''
	Specify columns of interest from each datasource and their directionality
	'''
	data_features = {
		'index_columns':	[ 'Country', 'ISO3' ],
		'inform_data':	{ # low is good
			'geopolitical_data': 	[ 'Human', 'Governance', 'Communication'], #, 'Institutional' ], # removed 'Institutional' for massive multicollinearity with the supply chain factors
			'environmental_data': 	[ 'Flood', 'Tsunami', 'Tropical Cyclone', 'Drought'	], # 'Drought' for multicoll with Flood (>.8)
			# 'vulnerability_data': 	[ 'Recent Shocks', 'Physical infrastructure' ] # removed 'Access to health care' for multicoll w/ Physical infrastructure (>.8)
			'vulnerability_data': 	[ 'Physical infrastructure', 'Development & Deprivation', 'Aid Dependency', 'DRR', 'Access to health care'] # removed 'Access to health care' for multicoll w/ Physical infrastructure (>.8)
		}, 
		'fm_global_data': { # high is good
			'environmental_data':	[ 'Exposure to Natural Hazard', 'Natural Hazard Risk Quality' ],
			'supply_chain_data':	[ 'Supply Chain Visibility', 'Quality of Infrastructure', 'Local Supplier Quality' ] # all massively multicoll
		},
		# 'fragile_states_data':	 'E1: Economy'  # removed for multicollinearity
		'fragile_states_data':	 ['P1: State Legitimacy', 'P3: Human Rights']  # low is good
	}

	###############
	
	df = pd.read_excel(Path.cwd() / 'INFORM_2019_v037.xlsx', sheet_name='INFORM 2019 (a-z)')
	df.columns = df.iloc[0]
	df = df.iloc[2:].reset_index(drop = True)
	df.rename(inplace = True, columns = {'COUNTRY': 'Country'})
	drop_subset = [x for x in list(df.columns) if str(x) not in data_features['index_columns']]
	df.dropna(inplace = True, how = 'all', subset = drop_subset)
	df.sort_values(by = 'Country', inplace = True); df.reset_index(inplace = True, drop = True)

	###############

	fsi_data = pd.read_excel(Path.cwd() / 'fsi-2018.xlsx', sheet_name = '2018')
	fsi_data = fsi_data[['Country'] + data_features['fragile_states_data']]
	fsi_data.sort_values(by = 'Country', inplace = True); fsi_data.reset_index(inplace = True, drop = True)

	###############

	fm_list = pd.read_excel(Path.cwd() / '2018ResilienceIndexRegions.xlsx', sheet_name = None)
	keys = list(fm_list.keys())
	main_sheet_name = [x for x in keys if str(x).count("Drivers") > 0][0]
	fm_data = fm_list[main_sheet_name]
	fm_data.dropna(axis = 0, how = 'all', inplace = True)
	fm_data.dropna(axis = 1, how = 'all', inplace = True)
	first_col = fm_data.columns[0]
	first_row = int(fm_data.loc[fm_data[first_col] == 'Country'].index[0])
	fm_data = fm_data.loc[first_row:]; fm_data.reset_index(inplace = True, drop = True)
	fm_data.columns = list(fm_data.loc[0].values); fm_data = fm_data.loc[1:]; fm_data.reset_index(inplace = True, drop = True)

	aggregation_cols = data_features['fm_global_data']['environmental_data'] + data_features['fm_global_data']['supply_chain_data']
	for col in aggregation_cols:
		fm_data[col] = fm_data[col].astype(float)

	fm_data['Country'] = fm_data['Country'].apply((lambda x : x.upper()))
	fm_data = aggregate_zones(fm_data, ['CHINA ZONE 1', 'CHINA ZONE 2', 'CHINA ZONE 3'], 'CHINA', aggregation_cols)
	fm_data = aggregate_zones(fm_data, ['UNITED STATES EAST', 'UNITED STATES CENTRAL', 'UNITED STATES WEST'], 'UNITED STATES OF AMERICA', aggregation_cols)

	###############

	### ----- NOTES ----- ###
	'''
	Clean up country names and then merge with inner joins
	'''
	for frame in [df, fm_data, fsi_data]:
		frame['Country'] = frame['Country'].apply((lambda x : clean_country_name(x)))
	scores = pd.merge(df, fm_data, how = 'inner', on = 'Country')
	scores = pd.merge(scores, fsi_data, how = 'inner', on = 'Country')
	scores.sort_values(inplace = True, by = 'Country')

	### ----- NOTES ----- ###
	'''
	Apply appropriate transformations for uniform scale across risk data
	'''
	for col in data_features['inform_data']['geopolitical_data']:	scores[col] = scores[col].apply((lambda x : 10 * x))
	for col in data_features['inform_data']['environmental_data']:	scores[col] = scores[col].apply((lambda x : 10 * x))
	for col in data_features['fragile_states_data']:	scores[col] = scores[col].apply((lambda x : 10 * x))
	for col in data_features['inform_data']['vulnerability_data']:	scores[col] = scores[col].apply((lambda x : 10 * x))
	for col in data_features['fm_global_data']['environmental_data']:	scores[col] = scores[col].apply((lambda x : 100 - x))
	for col in data_features['fm_global_data']['supply_chain_data']:	scores[col] = scores[col].apply((lambda x : 100 - x))


	### ----- NOTES ----- ###
	'''
	Apply composition methodology
	Subset on composite columns and reset index, then write to file
	'''

	########################

	risk_scores = scores.copy(deep = True)
	country_list = list(risk_scores['Country'].values)

	# # check if file exists?
	# get_world_bank_dataset('http://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.KD.ZG?downloadformat=excel', 'world_bank_pcap_gdp_growth')
	# get_world_bank_dataset('http://api.worldbank.org/v2/en/indicator/SL.UEM.TOTL.ZS?downloadformat=excel', 'world_bank_unem')
	# get_world_bank_dataset('http://api.worldbank.org/v2/en/indicator/NY.GDP.DEFL.KD.ZG?downloadformat=excel', 'world_bank_inflation')
	# get_world_bank_dataset('http://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.KD?downloadformat=excel', 'world_bank_gdp_pcap')

	df1 = pd.read_excel(Path.cwd() / 'world_bank_pcap_gdp_growth.xlsx')
	df2 = pd.read_excel(Path.cwd() / 'world_bank_unem.xlsx')
	df3 = pd.read_excel(Path.cwd() / 'world_bank_inflation.xlsx')
	df5 = pd.read_excel(Path.cwd() / 'world_bank_gdp_pcap.xlsx')

	df1 = prep_wb_data(df1, country_list)
	df2 = prep_wb_data(df2, country_list)
	df3 = prep_wb_data(df3, country_list)
	df5 = prep_wb_data(df5, country_list)


	##### create features
	df1['gdp_pcap_growth_volatility'] = df1.std(axis = 1)
	df2['unemployment'] = df2.mean(axis = 1)
	df3['inflation'] = df3.mean(axis = 1)
	df5['avg_gdp_pcap'] = -df5.mean(axis = 1)

	for frame in [df1, df2, df3, df5]:
		try:
			frame = frame[['Country', frame.columns[-1]]]
		except:
			print(frame.head())
		risk_scores = pd.merge(risk_scores, frame, how = 'left', on = 'Country')

	try:
		risk_scores.drop(['Unnamed: 0'], inplace = True, axis = 1)
	except:
		pass

	# 0-100 min-max scale the economic features
	for col in ['unemployment', 'inflation', 'gdp_pcap_growth_volatility', 'avg_gdp_pcap']:
		risk_scores[col] = 100*(risk_scores[col] - risk_scores[col].min())/(risk_scores[col].max() - risk_scores[col].min())


	####################################

	### identify column groups
	feature_hierarchy = {}
	feature_hierarchy['gp_risk'] = ['P1: State Legitimacy', 'Human']
	feature_hierarchy['ec_risk'] = ['avg_gdp_pcap', 'gdp_pcap_growth_volatility', 'unemployment', 'inflation']
	feature_hierarchy['nd_risk'] = [ 'Exposure to Natural Hazard', 'Aid Dependency'] + [ 'Flood', 'Tsunami', 'Tropical Cyclone'	, 'Drought']
	feature_hierarchy['sc_risk'] = [ 'Communication', 'Quality of Infrastructure']

	risk_cols = list(feature_hierarchy.keys())
	index_cols = ['Country', 'ISO3']

	def option_1(inframe, hierarchy, ind_cols):
		weighter = PCAWeighting(inframe, hierarchy, ind_cols)
		inframe = weighter.run()

		num_ranges = 10
		for col in risk_cols:
			new_col = col.replace('_risk', '_rating')
			inframe[new_col] = pd.Series(pd.qcut(inframe[col], num_ranges, labels = range(1, num_ranges + 1))).astype(int)

		inframe['ov_rating'] = inframe[[col.replace('_risk', '_rating') for col in risk_cols]].mean(axis = 1)

		inframe.to_csv(Path.cwd() / 'country_rankings_1.csv', index = False)

	option_1(risk_scores, feature_hierarchy, index_cols)
