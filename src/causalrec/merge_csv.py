import os
from fnmatch import fnmatch
import pandas as pd
import numpy as np

for idx in range(22):
	for gen in ['_wg', '_sg']:
		model = "simulation"+str(idx)+gen
		root = model + '_Yfit'
		pattern = 'res*bin0*.csv'

		print(model)

		filenames = []

		for path, subdirs, files in os.walk(root):
		    for name in files:
		        if fnmatch(name, pattern):
		            filenames.append(os.path.join(path, name)) 

		if len(filenames) > 0:

			combined_csv = pd.concat([pd.concat( [ pd.read_csv(f), pd.DataFrame(f.split('_')).T], axis=1)   for f in filenames] )

			combined_csv = combined_csv.sort_values("test_ndcg")


			combined_csv.to_csv(model + "_allres.csv", index=False)


			print(combined_csv["test_ndcg"].max(), combined_csv.shape)

			print(combined_csv[[fnmatch(model, "*wmf*") for model in combined_csv["model"]]].sort_values("test_ndcg")[["model", "alpha", "binary", "vad_normal_pos_plp", "vad_ndcg","test_ndcg"]][-1:])
			print(combined_csv[[fnmatch(model, "*pmf*") for model in combined_csv["model"]]].sort_values("test_ndcg")[["model", "alpha", "binary", "vad_normal_pos_plp", "vad_ndcg","test_ndcg"]][-1:])
			print(combined_csv[[fnmatch(model, "*pf*") for model in combined_csv["model"]]].sort_values("test_ndcg")[["model", "alpha", "binary", "vad_normal_pos_plp", "vad_ndcg","test_ndcg"]][-1:])


for model in ['coat_wg', 'coat_sg', 'R3_wg', 'R3_sg']:
	root = model + '_Yfit'
	pattern = 'res*bin0*.csv'

	print(model)

	filenames = []

	for path, subdirs, files in os.walk(root):
	    for name in files:
	        if fnmatch(name, pattern):
	            filenames.append(os.path.join(path, name)) 

	if len(filenames) > 0:

		combined_csv = pd.concat([pd.concat( [ pd.read_csv(f), pd.DataFrame(f.split('_')).T], axis=1)   for f in filenames] )

		combined_csv = combined_csv.sort_values("test_ndcg")


		combined_csv.to_csv(model + "_allres.csv", index=False)

		print(combined_csv["test_ndcg"].max(), combined_csv.shape)

		print(combined_csv[[fnmatch(model, "*wmf*") for model in combined_csv["model"]]].sort_values("test_ndcg")[["model", "alpha", "binary", "vad_normal_pos_plp", "vad_ndcg","test_ndcg"]][-1:])
		print(combined_csv[[fnmatch(model, "*pmf*") for model in combined_csv["model"]]].sort_values("test_ndcg")[["model", "alpha", "binary", "vad_normal_pos_plp", "vad_ndcg","test_ndcg"]][-1:])
		print(combined_csv[[fnmatch(model, "*pf*") for model in combined_csv["model"]]].sort_values("test_ndcg")[["model", "alpha", "binary", "vad_normal_pos_plp", "vad_ndcg","test_ndcg"]][-1:])



ress = []

for filenames in ["R3_wg_allres.csv", "coat_wg_allres.csv", "R3_sg_allres.csv", "coat_sg_allres.csv"]:
	file = pd.read_csv(filenames)

	print('\n\n\n'+filenames)

	print('\n select by ndcg \n')
	
	wmfdcf = file[[fnmatch(model, "*wmf_cau_*_add") for model in file["model"]]].sort_values("vad_ndcg100")[["model","test_ndcg100", "test_recall5"]].iloc[-1]
	wmfobs = file[[fnmatch(model, "*wmf_obs*") for model in file["model"]]].sort_values("vad_ndcg100")[["model","test_ndcg100", "test_recall5"]].iloc[-1]


	pmfdcf = file[[fnmatch(model, "*pmf_cau_*_add") for model in file["model"]]].sort_values("vad_ndcg100")[["model","test_ndcg100", "test_recall5"]].iloc[-1]
	pmfobs = file[[fnmatch(model, "*pmf_obs*") for model in file["model"]]].sort_values("vad_ndcg100")[["model","test_ndcg100", "test_recall5"]].iloc[-1]

	pfdcf = file[[fnmatch(model, "*pf_cau_*_add") for model in file["model"]]].sort_values("vad_ndcg100")[["model","test_ndcg100", "test_recall5"]].iloc[-1]
	pfobs = file[[fnmatch(model, "*pf_obs*") for model in file["model"]]].sort_values("vad_ndcg100")[["model","test_ndcg100", "test_recall5"]].iloc[-1]

	res = np.array([pmfobs, pmfdcf, pfobs, pfdcf, wmfobs, wmfdcf])

	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

	ress.append(np.array(res[:,1:],dtype=float))

	print(res)
	
	print(np.array(res[:,1:],dtype=float))


print("\n\nall res\n\n")

print(np.column_stack(ress))

# this reproduces Table 1 of Wang et al. (2020)