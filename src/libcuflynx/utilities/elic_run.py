# import opencor as oc
import numpy as np
from libcuflynx.solver_wrappers.opencor_helper import SimulationHelper
import csv

# TODO Finbar
# TODO Make this a simple example of running a cellml model that has been generated.

file_path = "/home/ash252/UoA/Autogen_git/circulatory_autogen/generated_models/cerebral_elic/cerebral_elic.cellml"
#file_path = "/home/ash252/UoA/RA/Friberg.cellml"
Outputfile_address = "/home/ash252/UoA/RA/"
Terminal_names_path = "/home/ash252/UoA/Finbar/"

names=[]

with open(Terminal_names_path + "Terminal names.csv") as file_name:
    file_read = csv.reader(file_name)

    array = list(file_read)

names=[]

for word in array:
    names.append(str(word))

for i in range(len(names)):
    names[i]=re.sub("_T/R_T", " ", names[i])
    names[i] = re.sub("\['", "", names[i])
    names[i] = re.sub("']", "", names[i])

#names = ['Friberg/Prol', 'Friberg/Tr1', 'Friberg/Tr2', 'Friberg/Tr3', 'Friberg/Circ'] , maximumStep=0.001

x = SimulationHelper(file_path, 0.001, 200, maximumNumberofSteps=1000, pre_time=0)
x.run()



input_variable=[]
for i in range(len(names)):
    print(names[i])
    input_variable.append(["R_T_" + names[i], x.get_results(array[i])[0, -1]])



with open (Outputfile_address+'elic_results.csv','w',newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ' ');
    my_writer.writerows(input_variable)




#y = x.get_results(['medial_occipital_occipitotemporal_branch_T52_L190_T/R_T'])[0, -1]
#print(y)



