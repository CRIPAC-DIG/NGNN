# dataset and preprocess

## brief introduction 

- polyvore/				the attribute of all the outfit and items

- polyvore_image/		the directory of outfit image, we give a small sample of an outfit (original dataset on 							Google Drive)
	/[outfitid]/0.jpg   the whole outfit performance
				1.jpg   the first items
				2.jpg   the second items

- polyvore_image_vec/	
	/[outfitid]\_[itemid].json

- polyvore_text/		the text information of all items

- polyvore_text_vec/	the vector of all items by Word2Vec

- polyvore_text_mhot/	the vector of all items by Muti-hot

- arranged_data/		the arranged data which is preprocessed for our code.

- use_inception.py      the file to extra the raw image of each item in `polyvore_image/` to vectors in 									`polyvore_image_vec` by inception-v3

- [preprocess.py]		stil in arranged.


## how to prepare for our model training





