import os

path = '/Users/ephraimsalhanick/Desktop/AzureMLSI2018/Gore_Images'

i = 0
for filename in os.listdir(path):
  os.rename(path+'/'+filename, path+'/gore_'+str(i)+'.jpg')
  i = i +1
