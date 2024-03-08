def interdict(dict1,dict2,separate = False):
	common_dict = {}
	for key in dict1.keys():
		if key in dict2.keys():
			common_dict[key] = dict1[key]
	
	if separate:
		other_dict = dict1 | dict2
		for key in common_dict.keys():
			if key in other_dict.keys():
				del other_dict[key]

	if separate:
		return common_dict, other_dict
	else:
		return common_dict

def listInterdict(lst_of_dicts):
	intersection = lst_of_dicts[0]
	for item in lst_of_dicts:
		intersection = interdict(intersection,item)

	return intersection