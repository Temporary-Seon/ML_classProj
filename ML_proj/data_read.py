# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 02:21:53 2019

@author: jo345
"""

import csv
import torch

#reading csv file math grade
f_math = open('data/student-mat.csv', 'r')
f_por = open('data/student-por.csv', 'r')
rdr_math = csv.reader(f_math)
rdr_por = csv.reader(f_por)

math_list = []
por_list = []
for line in rdr_math:
    
    #print(line)
    #break
    #print(len(line))
    # = 33 (quite many)
    math_list.append(line[1:10]+line[11:30])
    
#print(math_list[0])
#print(math_list[1])
    
#print(len(math_list))
print(math_list[0])

i = 0
for line in rdr_por:
    if i == 0:
        i = i+1
        continue
    else:
        por_list.append(line[1:10]+line[11:30])
        
#print(len(por_list))

#merging two lists
data_list = list(math_list[1:30])
data_list.extend(x for x in por_list[1:] if x not in data_list)

#print(data_list)
print('number of features :')
print(len(data_list[0]))
print(data_list[5])
print('data conversion')

#data modification letter => number
data = []
data_x = []
data_y = []
for row in data_list:
    temp_list = row
    #sex
    if row[0] == 'F':
        temp_list[0] = '0' #0 for female
    elif row[0] == 'M': 
        temp_list[0] = '1' #1 for male
    else:
        print('wrong index no at sex')
    
    #address
    if row[2] == 'U':
        temp_list[2] = '0' #0 for urban
    elif row[2] == 'R':
        temp_list[2] = '1' #1 for rural
    else:
        print('wrong index no at address')
    
    #family size
    if row[3] == 'GT3':
        temp_list[3] = '0' #0 for GT3
    elif row[3] == 'LE3':
        temp_list[3] = '1' #1 for LE3
    else:
        print('wrong index no at family size')
    
    #parent status
    if row[4] == 'T':
        temp_list[4] = '0' #0 for living together
    elif row[4] == 'A':
        temp_list[4] = '1' #1 for apart
    else:
        print('wrong index no at family size')
        
    #Mother's job
    if row[7] == 'teacher':
        temp_list[7] = '0' #0 for teacher
    elif row[7] == 'health':
        temp_list[7] = '1' #1 for health
    elif row[7] == 'services':
        temp_list[7] = '2' #2 for services
    elif row[7] == 'at_home':
        temp_list[7] = '3' #3 for at_home
    elif row[7] == 'other':
        temp_list[7] = '4' #4 for other

    #Father's job
    if row[8] == 'teacher':
        temp_list[8] = '0' #0 for teacher
    elif row[8] == 'health':
        temp_list[8] = '1' #1 for health
    elif row[8] == 'services':
        temp_list[8] = '2' #2 for services
    elif row[8] == 'at_home':
        temp_list[8] = '3' #3 for at_home
    elif row[8] == 'other':
        temp_list[8] = '4' #4 for other

    #student's guardian
    if row[9] == 'mother':
        temp_list[9] = '0' #0 for mother
    elif row[9] == 'father':
        temp_list[9] = '1' #1 for father
    elif row[9] == 'other':
        temp_list[9] = '2' #2 for other
    else:
        print('wrong index no at guardian')
        
    if row[13] == 'yes':
        temp_list[13] = '1' #1 for yes
    elif row[13] == 'no':
        temp_list[13] = '0' #0 for no
        
    if row[14] == 'yes':
        temp_list[14] = '1' #1 for yes
    elif row[14] == 'no':
        temp_list[14] = '0' #0 for no
        
    if row[15] == 'yes':
        temp_list[15] = '1' #1 for yes
    elif row[15] == 'no':
        temp_list[15] = '0' #0 for no
        
    if row[16] == 'yes':
        temp_list[16] = '1' #1 for yes
    elif row[16] == 'no':
        temp_list[16] = '0' #0 for no
        
    if row[17] == 'yes':
        temp_list[17] = '1' #1 for yes
    elif row[17] == 'no':
        temp_list[17] = '0' #0 for no
        
    if row[18] == 'yes':
        temp_list[18] = '1' #1 for yes
    elif row[18] == 'no':
        temp_list[18] = '0' #0 for no
        
    if row[16] == 'yes':
        temp_list[16] = '1' #1 for yes
    elif row[16] == 'no':
        temp_list[16] = '0' #0 for no
    
    if row[19] == 'yes':
        temp_list[19] = '1' #1 for yes
    elif row[19] == 'no':
        temp_list[19] = '0' #0 for no
    
    if row[20] == 'yes':
        temp_list[20] = '1' #1 for yes
    elif row[20] == 'no':
        temp_list[20] = '0' #0 for no
    
    temp_list_i = [int (i) for i in temp_list]
    data.append(temp_list_i)
    
    #x and y
    temp_list_x = temp_list_i[:24] + temp_list_i[26:]
    temp_list_y = temp_list_i[24:26]

    data_x.append(temp_list_x)
    data_y.append(temp_list_y)

print(data[5])
print(data_x[5])
print(data_y[5])
print('number of data :')
print(len(data))

#from list to matrix
data_x_tensor = torch.tensor(data_x)
data_y_tensor = torch.tensor(data_y)
#print(torch.is_tensor(data_tensor))
#print(data_tensor)
torch.save(data_x_tensor, 'tensorx.pt')
torch.save(data_y_tensor, 'tensory.pt')

f_math.close()
f_por.close()