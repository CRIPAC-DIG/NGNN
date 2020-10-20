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

- category_id.txt	the id to category name index for all items

- use_inception.py      the file to extra the raw image of each item in `polyvore_image/` to vectors in 									`polyvore_image_vec` by inception-v3

- [preprocess.py]		stil in arranged.


## how to prepare for our model training
First:              download and unzip the polyvore-images from Google Drive [here](https://drive.google.com/drive/folders/0B4Eo9mft9jwoVDNEWlhEbUNUSE0)  

Second:             use inception-v3 to extract image feature into `polyvore_image_vec/`
                    `python use_inception_for_vec.py`, click [here](https://drive.google.com/open?id=1ibYEw0H9L9O9OLbxCiAlcZkt_IYuwKfd) for the extracted feature with the same format.

Thrid:              download and unzip the detail informations of polyvore outfit. You can download the [original version](https://drive.google.com/drive/folders/0B4Eo9mft9jwoVDNEWlhEbUNUSE0)
or our [filtered verison](https://drive.google.com/open?id=1ibYEw0H9L9O9OLbxCiAlcZkt_IYuwKfd)

Fourth:		    generate `polyvore_text_mhot` file by `onehot_embedding.py`

Fiveth:		    `python summarize.py` filter the category which appears less than 100 times.

Sixth:		    `python preprocess.py` filter the items according to our paper. (this file is missing unfortunately, we upload the filtered dataset for you in `.\data`, if you don't want to write it by yourself.)










