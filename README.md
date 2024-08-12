**FCN**: 

1. Place dataset folders (benign, malignant) in results folder where FCN code files are kept.
2. Execute the FCN_compress_train.py file which will train and generate a weights file.
3. Execute FCN_test_.plot.all.py which will generate predicted images for all test scans.
4. Execute FCN_compress_genetic_train.py and redirect output to a text file. It will generate 11 weight files.
5. Execute FCN_compress.test_plot.all.py by getting the reduced number of filters for corresponding compressed weight file from output txt file of step 4 and get all predicted test scans outputs.

**UNet**:

1. Place dataset folders (benign, malignant) in results folder where UNet code files are kept.
2. Execute the UNet_gray_mini.train.py file which will train and generate a weights file.
3. Execute UNet_gray_mini.test.plot.all.py which will generate predicted images for all test scans.
4. Execute UNet_gray_compress_genetic.train.py and redirect output to a text file. It will generate 11 weight files.
5. Execute UNet_gray_compress.test.plot.all.py by getting the reduced number of filters for corresponding compressed weight file from output txt file of step 4 and get all predicted test scans outputs.

**Mini SegNet**:

1. Place dataset folders (benign, malignant) in results folder where Mini SegNet code files are kept.
2. Execute the SegNet_mini_train.train.py file which will train and generate a weights file.
3. Execute SegNet_mini_test.plot.all.py which will generate predicted images for all test scans.
4. Execute SegNet_mini_compress_genetic.train.py and redirect output to a text file. It will generate 11 weight files.
5. Execute SegNet_mini_compress.test.plot.all.py by getting the reduced number of filters for corresponding compressed weight file from output txt file of step 4 and get all predicted test scans outputs.


