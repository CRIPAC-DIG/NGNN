# dataset and preprocess

## brief introduction 

- polyvore/   			-- the attribute of all the outfit and items

- polyvore-images/ 		-- the directory of outfit image, we give a small sample of an outfit (original dataset on 							Google Drive)
	/[outfitid]/0.jpg 	-- the whole outfit performance
				1.jpg   the first items
				2.jpg   the second items

- polyvore_image_vectors/	
	/[outfitid]\_[itemid].json

- polyvore_text/		the text information of all items

- polyvore_text_vec/	the vector of all items by Word2Vec

- polyvore_text_mhot/	the vector of all items by Muti-hot

- arranged_data/		the arranged data which is preprocessed for our code.

- use_inception.py      the file to extra the raw image of each item in `polyvore_image/` to vectors in 									`polyvore_image_vec` by inception-v3

- [preprocess.py]		stil in arranged.


## how to prepare for our model training
First:              download and unzip the polyvore-images from Google Drive [here](https://drive.google.com/drive/folders/0B4Eo9mft9jwoVDNEWlhEbUNUSE0)  

Second:             use inception-v3 to extract image feature into `polyvore_image_vec/`
                    `python use_inception_for_vec.py`

Thrid:              download the detail informations of polyvore outfit. You can download the [original version](https://drive.google.com/drive/folders/0B4Eo9mft9jwoVDNEWlhEbUNUSE0)
or our [filtered verison](http:/xxx.com)





