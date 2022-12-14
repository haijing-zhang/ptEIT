In the "Data" folder, "random_200" and "random_500" stores the processed dataset based on 200 and 500 point resolution. "raw_data" folder
stores raw parameter used for generating point cloud and corresponding voltage measurements. "dataprocessing.py" is the file generating 
point cloud from parameters and creating dataset. "sensor_208.py" stores the coordinates of the electrodes.

ptEIT reconstructs the objects' conductivity, shape and position independently. "conduct_model.py" and "conduct_train.py" stores the
model and training file for predicting the amount of object in each image and their conductivity level. "pos_model.py" and "pos_train.py" 
are the model file and training file for predicting each object's center's coordinates."shape_model.py" and "shape_train.py" are the files 
reconstructing the objects' shapes.

The final trained models are stored in the "saved_Models" file. To test the final trained model on the testing dataset, firstly run the
"PLOTRESULT.py" file, which will show the reconstructed image on one random selected test sample and evaluate the performance metric on 
the whole testing dataset. The random selected test result would also be automatically stored in the "matplot" folder, which used for 
plotting the results in MATLAB. To do that, run the "simulation_plot.m" file in the "matplot" folder to show the MATLAB plots. 
