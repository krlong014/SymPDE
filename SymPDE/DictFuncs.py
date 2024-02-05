def interdict(dict1,dict2):
	new_dict = {}
	for key in dict1.keys():
		if key in dict2.keys():
			new_dict[key] = dict1[key]

	return new_dict