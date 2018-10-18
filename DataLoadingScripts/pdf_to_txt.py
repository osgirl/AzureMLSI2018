import pdftotext
import os

path = '/Users/ephraimsalhanick/Desktop/AzureMLSI2018/Bush_Speeches'

for filename in os.listdir(path):
	
	with open(path + '/' + filename, "rb") as f:
    		pdf = pdftotext.PDF(f)
	
	text = "\n\n".join(pdf)
		
	with open(path + '/' + filename.replace('pdf','txt'), "w") as file_out:
		file_out.write(text)
			
