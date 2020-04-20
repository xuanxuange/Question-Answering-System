# pip3 install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_lg-3.0.0/en_coref_lg-3.0.0.tar.gz

# pip3 install stanfordcorenlp

def get_sent_num(thresholds_list, start):
	print(thresholds_list)
	if start >= thresholds_list[-1]:
		return len(thresholds_list) - 1

	loind = 0
	hiind = len(thresholds_list) - 2

	while loind < hiind:
		loval = thresholds_list[loind]
		hival = thresholds_list[hiind + 1]
		approx_size = (hival - loval) // (hiind + 1 - loind)
		hyp = ((start - loval) // approx_size) + loind
		print(str(loval) + ", " + str(hival) + ", " + str(approx_size) + ", " + str(thresholds_list[hyp]) + ", " + str(start))

		if start >= thresholds_list[hyp]:
			if start < thresholds_list[hyp + 1]:
				hiind = hyp
				loind = hyp
			else:
				loind = hyp + 1
		else:
			if start >= thresholds_list[hyp - 1]:
				loind = hyp - 1
				hiind = hyp - 1
			else:
				hiind = hyp - 2
	print("Found <" + str(start) + "> in bucket [" + str(thresholds_list[loind]) + ", " + str(thresholds_list[loind + 1]) + ")")

	return loind