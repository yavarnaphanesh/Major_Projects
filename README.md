# Major_Projects
AUTOMATING E-GOVERNMENT SERVICES WITH MACHINE LEARNING AND ARTIFICIAL INTELLIGENCE
In this paper author describing concept to predict insurance policies charges and user opinion sentiment on policies by applying machine learning and artificial intelligence. Machine learning can automatically predict future values by analysing past historic data and artificial intelligence will take decision as human brain (as our brain help us in making decision as word hard if marks is less else take easy). Here also by analysing male and female BMI index AI and machine learning will predict insurance policy and its charges. This AI and ML can also analyse user’s opinions or reviews and then it will take decision as whether person’s opinion is positive, negative or neutral. 
To implement this project we are using insurance dataset from KAGGLE website from below URL
https://www.kaggle.com/mirichoi0218/insurance
Below screen shots showing details of dataset
 
In above dataset screen shots first row contains dataset column names and other rows contains dataset value and in above dataset we have age, gender, BMI and smoking and other attributes and in last column we have policy charges and by using above dataset we will train AI and ML algorithm and then when we input new test data then ML will predict charges for each test record and in below screen we can see some test records
 
In above test data we don’t have charges column and when we apply above data on ML then ML will predict charges for above test data.
SCREEN SHOTS
To run project double click on ‘run.bat’ file to get below screen
 
In above screen click on ‘Upload Insurance Dataset’ button and upload dataset
 
In above screen selecting and uploading “insurance.csv” file and the click on ‘Open’ button to load dataset and to get below screen
 
In above screen dataset loaded and we can see dataset contains some non-numeric values and ML will not take string value so we need to convert non-numeric string values to numeric by replacing male with 0 and female with 1. So click on ‘Explore Insurance Dataset’ button to replace string with numeric values
 
In above screen we can see all string values replace with numeric data and we can see dataset contains total 1338 records and application using 1070 records to train ML and 268 records to test ML performance and in above graph x-axis represents AGE and y-axis represents insurance charges and we can see in above graph when AGE increasing then insurance charges also increasing. Now dataset is ready and now click on ‘Run Machine Learning Algorithm’ button to build ML model
 
In above screen ML model generated and now click on ‘Predict BMI Based Insurance Charges’ button to upload test data and then ML will predict insurance policy and charges 
 
In above screen selecting and uploading ‘test.csv’ file and then click on ‘Open’ button to load test records and then ML and AI will analyse gender, smoking, BMI and then predict policy charges and will give below output
 
In above screen in square bracket we can see test values and after square bracket we can see predicted policy charges. After seeing charges and policy user may express some opinion and then ML will predict sentiment from that opinion. So click on ‘Predict Sentiments on Insurance’ button to express some reviews
 
In above screen in dialog box I entered some review and then click on ‘OK’ button to get below result
 
In above screen for given review AI and ML predict sentiment as POSITIVE. Similarly you can enter any opinion and then ML will predict sentiment 
