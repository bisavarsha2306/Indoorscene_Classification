import splitfolders

input_folder= r'D:\mohanteja\python_files\dataset'
#splitting data by ratios i.e, into 10% nd 90% as test nd training data rspectively
splitfolders.ratio(input_folder, output='D:/mohanteja/python_files/dataset_1',
                    seed=42, ratio=(.7, .2, .1),
                    group_prefix=None)
#splitting images using fixed values like 10 for training nd 50 for tesing                     
'''splifolders.fixed(input_folder, output='D:/mohanteja/python_files/dataset_1',
                   seed=42, fixed=(100,100),
              oversample=True, group_prefix=None)'''