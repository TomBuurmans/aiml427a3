Read Me File
For Both Program same steps will work. Please change the program name that you like to run in the cluster.

	1.	Create a folder and save the java code file, SetupSparkClasspath.csh and hadoop_2.8.0_jars_files.

Please download the setup files from this link:

	a)	Java Files: 
https://gitlab.ecs.vuw.ac.nz/buurmatom/aiml427a3

	b)	Data File
https://ecs.wgtn.ac.nz/foswiki/pub/Courses/AIML427_2021T1/Assignments/kdd.data

	c)	Hadoop 2.8.0 Jars File 
https://ecs.wgtn.ac.nz/foswiki/pub/Courses/AIML427_2021T1/Assignments/hadoop_2.8.0_jars_files.zip

	d)	SetupSparkClasspath.csh https://ecs.wgtn.ac.nz/foswiki/pub/Courses/AIML427_2021T1/Assignments/SetupSparkClasspath.csh

	e)	Putty
             https://the.earth.li/~sgtatham/putty/latest/w64/putty-64bit-0.75-installer.msi


	2.	Download WinSCP and connect to ECS account to file transfer using SFTP or SCP. 

Please download the setup files from this link:

	a)	WinSCP setup File 
https://winscp.net/eng/download.php

             Transfer the file on the ECS account. 
           

	3.	Open the Terminal and connect to your ECS account. 
	4.	Set a couple of environmental variable by writing following commands. 

	a)	Source foldername/SetupSparkClasspath.csh
	b)	Source foldername/SetupHadoopClasspath.csh

	5.	Access the cluster code using the following command: 
		%ssh co246a-1

	6.	Now set up the environmental variables for the Hadoop Cluster. 
	a)	Source foldername/SetupSparkClasspath.csh
	b)	Source foldername/SetupHadoopClasspath.csh

	7.	Now move the data file from the Local File System to the HDFS 
	8.	Now go to the directory containing code. 
	9.	Make a new folder e.g. Finalwork
	10.	Run the following command (Please replace the modelname with the program that you like run. 

	a)	 Javac -cp “jars/*” -d  modelname.java
	b) Jar cvf modelname.jar -C Finalwork

	11.	Now deploy it to the cluster using the following command: 
	
	      Spark-submit –class “modelname” –master yarn -deplay-mode cluster model.jar /user/ecsuser/kdd.data /user/ecsuser/output

	12.	Check the output using
	
	     hdfs dfs -cat /user/ecsuser/output/part-00001
			
