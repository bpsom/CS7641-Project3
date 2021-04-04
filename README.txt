Project3 - Clustering and Dimensionality Reduction
Bommegowda P. Somashekharagowda
GTID: 903387896 
CS7641
Spring-2021

1) Code github Repo link -
code github link:
	https://github.gatech.edu/bps6/CS7641-Project3
	https://github.com/bpsom/CS7641-Project3
	The repo consists of all source files to execute the project, datasets
	github (gatech) last commit id: ab34e0750689ead6490b29b2f939f18852e54ce4
	Note that after this commit only Reame.txt file was modified to include this commit information.

2) File Details
	The code has below source files -
		ULTestLearners.py
		ULClustering.py
		ULDimensionReduction.py
		SUNeuralNetworks.py
	Dataset files -
		Data\winequality-white.csv

	results directory consists of the plots and the result.txt file when the code is executed -
		result\
		result\wine-Q
		result\digits
		result\Results.txt

	
3) Required Libraries to execute the code
	env.yml consists of packages that are required to execute the code.
    On system terminal where Anaconda is installed (supporting python 3.6) execute - 
		$conda env create --file env.yml
    
	The main libraries required to execute the code on python3.6 are scikit-learn, numpy, pandas, matplotlib
	
    Then enter the env to execute the code as in Code execution
		$conda activate ml
    
    To exit the env 
		$conda deactivate	


4) Working environment where this code was used (System Setup)

    On a Windows 10 machine Pycharm editor with Anaconda python 3.6 as project interpreter was used to implement the code. 
	

5) Code execution

    On a python3.6 console execute command with all required libraries installed, activate conda environment as explained above.
    Then execute `ULTestLearners.py`

	For Linux environment
		$ python ULTestLearners.py
	
    Note: Code execution takes ~700seconds to produce the results. 
    
6) Output

    The code generates plots and a Results.txt file with data. They are stored into `result` directory.
	
