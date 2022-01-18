# HW2 ― GAN and UDA
In this assignment, you are given datasets of human face and digit images. You will need to implement the models of both GAN and ACGAN for generating human face images and digits, respectively, and the model of DANN for classifying digit images from different domains.

<p align="center">
  <img width="853" height="500" src="http://mmlab.ie.cuhk.edu.hk/projects/CelebA/intro.png">
</p>

For more details, please click [this link](https://drive.google.com/drive/folders/1loYdSncANJHv9qtIcb5Dsmp0ImcdIPn4?usp=sharing) to view the slides of HW2. **Note that all of hw2 videos and introduction pdf files can be accessed in your NTU COOL.**

# Usage
To start working on this assignment, you should clone this repository into your local machine by using the following command.

    git clone https://github.com/DLCV-Fall-2021/hw2-<username>.git
Note that you should replace `<username>` with your own GitHub username.

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `hw2_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1BwZiFfGKAqIOFRupt6xO7-KuhPYd5VMO/view?usp=sharing) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `hw2_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

### Evaluation
To evaluate your UDA models in Problems 3 and Bonus, you can run the evaluation script provided in the starter code by using the following command.

    python3 hw2_eval.py $1 $2

 - `$1` is the path to your predicted results (e.g. `hw2_data/digits/mnistm/test_pred.csv`)
 - `$2` is the path to the ground truth (e.g. `hw2_data/digits/mnistm/test.csv`)

Note that for `hw2_eval.py` to work, your predicted `.csv` files should have the same format as the ground truth files we provided in the dataset as shown below.

| image_name | label |
|:----------:|:-----:|
| 00000.png  | 4     |
| 00001.png  | 3     |
| 00002.png  | 5     |
| ...        | ...   |

# Submission Rules
### Deadline
110/11/23 (Tue.) 03:00 AM (GMT+8)

### Late Submission Policy
You have up to 3 free late days quota depending on your hw0 result. Once you have exceeded your quota, the credit of any late submission will be deducted by 30% each day.

Note that while it is possible to continue your work in this repository after the deadline, **we will by default grade your last commit before the deadline** specified above. If you wish to use your quota or submit an earlier version of your repository, please contact the TAs and let them know which commit to grade.

### Academic Honesty
-   Taking any unfair advantages over other class members (or letting anyone do so) is strictly prohibited. Violating university policy would result in an **F** grade for this course (**NOT** negotiable).    
-   If you refer to some parts of the public code, you are required to specify the references in your report (e.g. URL to GitHub repositories).      
-   You are encouraged to discuss homework assignments with your fellow class members, but you must complete the assignment by yourself. TAs will compare the similarity of everyone’s submission. Any form of cheating or plagiarism will not be tolerated and will also result in an **F** grade for students with such misconduct.

### Submission Format
Aside from your own Python scripts and model files, you should make sure that your submission includes *at least* the following files in the root directory of this repository:
 1.   `hw2_<StudentID>.pdf`  
The report of your homework assignment. Refer to the "*Grading*" section in the slides for what you should include in the report. Note that you should replace `<StudentID>` with your student ID, **NOT** your GitHub username.
 2.   `hw2_p1.sh`  
The shell script file for running your GAN model. This script takes as input a path and should output your 1000 generated images in the given path.
 3.   `hw2_p2.sh`  
The shell script file for running your ACGAN model. This script takes as input a path and should output your 1000 generated images in the given path.
 4.   `hw2_p3.sh`  
The shell script file for running your DANN model. This script takes as input a folder containing testing images and a string indicating the target domain, and should output the predicted results in a `.csv` file.
 5.   `hw2_bonus.sh`  
The shell script file for running your improved UDA model. This script takes as input a folder containing testing images and a string indicating the target domain, and should output the predicted results in a `.csv` file.

We will run your code in the following manner:

    bash ./hw2_p1.sh $1
    bash ./hw2_p2.sh $1
    bash ./hw2_p3.sh $2 $3 $4
    bash ./hw2_bonus.sh $2 $3 $4

-   `$1` is the path to your output generated images (e.g. `~/hw2/GAN/output_images` or `~/hw2/ACGAN/output_images`).
-   `$2` is the directory of testing images in the **target** domain (e.g. `~/hw2_data/digits/mnistm/test`).
-   `$3` is a string that indicates the name of the target domain, which will be either `mnistm`, `usps` or `svhn`. 
	- Note that you should run the model whose *target* domain corresponds with `$3`. For example, when `$3` is `mnistm`, you should make your prediction using your "SVHN→MNIST-M" model, **NOT** your "MNIST-M→SVHN" model.
-   `$4` is the path to your output prediction file (e.g. `~/test_pred.csv`).

> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

### Packages
This homework should be done using python 3.8. For a list of packages you are allowed to import in this assignment, please refer to the requirments.txt for more details.

You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

### Remarks
- If your model is larger than GitHub’s maximum capacity (100MB), you can upload your model to another cloud service (e.g. Dropbox). However, your shell script files should be able to download the model automatically. For a tutorial on how to do this using Dropbox, please click [this link](https://goo.gl/XvCaLR).
- **DO NOT** hard code any path in your file or script, and the execution time of your testing code should not exceed an allowed maximum of **10 minutes**.
- **Please refer to HW2 slides for details about the penalty that may incur if we fail to run your code or reproduce your results.**

# Q&A
If you have any problems related to HW2, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under hw2 FAQ section in FB group
