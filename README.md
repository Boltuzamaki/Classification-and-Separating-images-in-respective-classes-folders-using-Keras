# Classification using keras and VGG16 and seperate classified image into respective classes subfolders

If you have a test directory which contains random images in same directory and you want to predict test image and move them in folders according to prediction then you are at right place just follow the steps and you will get what you want. 

# Folders and file structure of test and train set
## Working directory
- **flow**
  -  train
     - class1   
       - image1.jpg
       - image2.jpg
       - ......
     - class2
       - image1.jpg
       - image2.jpg
       - ......
    
     - ...
   - test
      - image1.jpg
      - image2.jpg
      - .....
      
The train folder may contains as many classes as you want and test contains only images because we want to predict them.

Then open test folder and select all image by 'ctrl+A' and then press F2 and rename 1st image as 'a'.Then your test folder looks like this

- **test**
   - a (1).jpg
   - a (2).jpg
   - a (3).jpg
   - ...
   
# Edit the code where necessary

Now open the classification.py

I made comment like '#### 1','#### 2 ' .. where you have to make changes and I also mentioned below what changes you have to made.

 - '#### 1'  -- Train folder path
 - '#### 2'  -- Test folder path
 - '#### 3' -- Validate folder path
 - '#### 4'  -- Classes name replace it same as folder names in train folder
 - '#### 5'  -- Same as 4
 - '#### 6'  -- Same as 4
 - '#### 7'  -- Classes number and image dimension
 - '#### 8' -- Classes number
 - '#### 9' -- Number of images in test folder like if you have 20 image then change (1,21),ie. last one 20+1 = 21
 - '#### 10' -- Test image dimension
 - '#### 11' -- Number of class like if you have 5 classes then (0,6),ie. last one 5+1 = 6
 
# Run the edited file 

    
    

