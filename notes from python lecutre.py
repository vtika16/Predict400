##arrays are intended just to hold numeric values, they keep the values next to each other. makes calcs on entire arrays quicker than lists
#arrays can only call numeric data types
#they can be any dimension you want, arrays can hold arrays. crazy dimensional representation of the data
#makes it diff from matrix since its only a two dimensional array
#

print("hey bitch")
I = int(5)

print(I)

import csv
import numpy

with open('Desktop/train.csv') as f: ##open file in context manager
    csv_reader = csv.DictReader(f) ##create csv
    contents = list(csv_reader) ##bring it into memory
    '''
    ##same as contents = list()
    contents = []
    for row in csv_reader:
        #do process
        contents.append(row)
    '''
print('left context')
print(contents[0]) #prints first row of csv that is brought in

updated_contents = [row for row in contents if row['Sex'] == 'male'] ##just get the male rows

# open the file to write out the new rows
with open('Desktop/train_men.csv', 'w') as f:
    headerrow = updated_contents[0].keys()
    csv_writer = csv.DictWriter(f, fieldnames=headerrow)
    csv_writer.writeheader()
    csv_writer.writerows(updated_conents)
    
print("done writing")

    