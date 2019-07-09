#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class PCAWeighting:

	def __init__(self, df, feature_hierarchy, index_cols):
		self.hierarchy = feature_hierarchy
		self.orig_data = df
		self.index_cols = index_cols # maybe get via elimination from features in the feature hierarchy
		self.feature_cols = list(feature_hierarchy.keys())
		self.data = None
		self.weights = None
		self.overall_weights = None
		self.pca = None

	@staticmethod
	def varimax(phi, gamma = 1.0, q = 20, tol = 1e-6):
		"""
		Stolen from the internet somewhere
		"""
		p, k = phi.shape
		R = np.eye(k)
		d = 0
		for i in range(q):
			d_old = d
			Lambda = np.dot(phi, R)
			u, s, vh = np.linalg.svd(np.dot(phi.T, np.asarray(Lambda)**3 - (gamma / p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
			R = np.dot(u, vh)
			d = np.sum(s)
			if d_old != 0 and d/d_old < 1 + tol: break
		return np.dot(phi, R)

	@staticmethod
	def apply_threshold(inval, thresh):
		if inval > thresh:
			return inval
		else:
			return 0

	def fit_pca(self, inframe, feature_list, eigenvalue_floor = 0.8, exp_var_floor = 0.1):
		pca = PCA(n_components = len(feature_list)) ### standard scale, then back to regular units?
		pca.fit(inframe[feature_list])

		print('\n\n')
		print(inframe[feature_list].corr()) # strong correlations

		# eigenval = [eigenval for eigenval in pca.explained_variance_ if eigenval > 1]
		# I was told these ^^^ were the eigenvalues...
		x_std = StandardScaler().fit_transform(inframe[feature_list])
		cov_mat = np.cov(x_std.T)
		eig_vals, eig_vecs = np.linalg.eig(cov_mat)
		# print(eig_vals)
		eig_vals = [x for x in sorted(eig_vals) if x > eigenvalue_floor]

		ten_perc = [exp_var for exp_var in pca.explained_variance_ratio_ if exp_var > exp_var_floor]
		keepers = min(len(ten_perc), len(eig_vals))
		if keepers != len(feature_list):
			pca = PCA(n_components = keepers)
			pca.fit(inframe[feature_list])
			print("Explained variance:  {:0.1f}%".format(np.sum(pca.explained_variance_ratio_)*100))
			if np.sum(pca.explained_variance_ratio_)*100 < 0.6:
				print("\n\n*****WARNING:  Less than 60% of variance explained by the principal components.\n\n")
		# print("Exp var Keepers:  " + str(len(ten_perc)))
		# print("EIG keepers:  " + str(len(eig_vals)))
		# print(eig_vals)
		# print("EIG keepers:  " + str(len([x for x in eigenval if x > 1])))
		self.pca = pca

	def remove_weights_below_threshold(self, threshold = .01):
		weights = self.weights
		weights_w_floor = {k:self.apply_threshold(v, threshold) for k, v in weights.items()}
		if sorted(list(weights_w_floor.values())) != sorted(list(weights.values())):
			new_total = np.sum(list(weights_w_floor.values()))
			self.weights = {k:v/new_total for k, v in weights_w_floor.items()}

	def get_init_weights(self, feature_list):
		self.feature_list = feature_list
		self.fit_pca(self.data, self.feature_list)
		result = self.varimax(self.pca.components_)
		weights_df = pd.DataFrame()
		for i, factor in enumerate(result):
			feature_weights = {}
			for j, feature in enumerate(factor):
				feature_weights[self.feature_list[j]] = feature**2
			test_df = pd.io.json.json_normalize(feature_weights)
			test_df.index = ['PCA' + str(i+1)]
			weights_df = pd.concat([weights_df, test_df], axis = 0)
		self.weights_df = weights_df.T

	def get_intermediate_composites(self):
		# get intermediate composites
		df = self.weights_df.copy(deep = True)
		for col in self.weights_df.columns:
			df[col] = (self.weights_df[col] == self.weights_df.max(axis = 1))
		composites = {}
		for col in df.columns:
			trues = list(df[df[col] == True].index.values)
			composites[col] = trues
		del df
		self.composites = composites

	def rescale_explained_variances(self):
		# rescale explained variance
		variances = list(self.pca.explained_variance_ratio_)
		if len(self.feature_list) != len(self.weights_df.columns):
			variances = [x for i, x in enumerate(variances) if i in range(len(self.weights_df.columns))]
			variances = [x/np.sum(variances) for x in variances]
		factor_weights = {}
		for i, col in enumerate(self.weights_df.columns):
			factor_weights[col] = variances[i]
		self.factor_weights = factor_weights

	def get_final_weights(self):
		# get denominator and final weights
		total = 0; weights = {}
		for k, v in self.composites.items():
			for item in v:
				total += self.weights_df.loc[item][k]*self.factor_weights[k]
				weights[item]=self.weights_df.loc[item][k]*self.factor_weights[k]
		weights = {k:v/total for k, v in weights.items()}
		# del weights_df
		assert sum(weights.values()).round(5) == 1.0, 'Weights sum to {:0.5f}'.format(sum(weights.values()))

		"""
		## old, wrong way to do it
		old_weights = {}
		try:
			for factor in result:
				factor = [round(abs(x), 5) for x in factor]
				feature = factor.index(float(1))
				old_weights[feature_list[feature]] = pca.explained_variance_ratio_[feature]
		except:
			for i, factor in enumerate(result[0]):
				old_weights[feature_list[i]] = factor**2
		assert sum(old_weights.values()).round(5) == 1.0, 'Weights sum to {:0.5f}'.format(sum(old_weights.values()))
		print(old_weights)
		print('\n\n\n\n')
		"""

		self.weights = weights
		self.remove_weights_below_threshold()
		print('\n\n')
		print(self.weights)
		print('\n\n')
		return self.weights

	def get_weights(self, features):
		self.get_init_weights(features)
		self.get_intermediate_composites()
		self.rescale_explained_variances()
		return self.get_final_weights()

	@staticmethod
	def make_component_scores(indict, inframe):
		warpframe = inframe.copy(deep = True)
		for key in list(indict.keys()):
			subdict = indict[key]
			for k, v in subdict.items():
				warpframe[k] = warpframe[k].apply((lambda x : x*v))
			inframe[key] = warpframe[list(subdict.keys())].sum(axis=1)
		return inframe

	def run(self):
		# subset data
		collist = self.index_cols
		for col in self.feature_cols:
			collist += self.hierarchy[col]
		self.data = self.orig_data[collist]

		# get component weights
		component_weights = {x: self.get_weights(self.hierarchy[x]) for x in self.feature_cols}
		self.data = self.make_component_scores(component_weights, self.data)
		
		# subset data
		assert len(self.data.columns) == len(self.index_cols) + len(self.feature_cols), 'Missing features!'

		# get overall weights
		overall_weights = self.get_weights(self.feature_cols)
		self.overall_weights = {'overall_weights': overall_weights}
		self.data = self.make_component_scores(self.overall_weights, self.data)
		# self.data.to_csv('class_test.csv', index = False)
		return self.data
